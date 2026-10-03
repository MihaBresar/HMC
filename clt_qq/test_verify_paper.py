"""Catch corruption and mismatched provenance in the published paper bundle."""

import contextlib
import csv
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from .verify_paper import PAPER_DIR, main, verify_paper


class VerifyPaperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.paper = Path(self.temp.name) / "paper"
        for directory in ("data_80k", "figures", "latex"):
            shutil.copytree(PAPER_DIR / directory, self.paper / directory)

    def edit_json(self, relative, edit):
        path = self.paper / relative
        data = json.loads(path.read_text())
        edit(data)
        path.write_text(json.dumps(data))

    def edit_csv(self, relative, edit):
        path = self.paper / relative
        with path.open(newline="") as stream:
            rows = list(csv.reader(stream))
        edit(rows)
        with path.open("w", newline="") as stream:
            csv.writer(stream).writerows(rows)

    def test_published_bundle_and_cli_pass_without_sampling(self):
        self.assertIn("24,000 chain averages", verify_paper(self.paper))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["--paper-dir", str(self.paper)]), 0)
        self.assertIn("No simulations were run", output.getvalue())

    def test_changed_protocol_is_rejected(self):
        self.edit_json("data_80k/t3_abs/config.json", lambda x: x.update(burn_in=10000))
        with self.assertRaisesRegex(ValueError, "config burn_in"):
            verify_paper(self.paper)

    def test_reordered_csv_chains_are_rejected(self):
        def swap(rows):
            rows[1], rows[2] = rows[2], rows[1]
        self.edit_csv("data_80k/t1_tail/chain_means_t1_tail.csv", swap)
        with self.assertRaisesRegex(ValueError, "CSV chain IDs"):
            verify_paper(self.paper)

    def test_csv_value_change_is_rejected(self):
        self.edit_csv("data_80k/t1_tail/chain_means_t1_tail.csv",
                      lambda rows: rows[1].__setitem__(1, "0.25"))
        with self.assertRaisesRegex(ValueError, "CSV chain averages differ"):
            verify_paper(self.paper)

    def test_nonfinite_or_wrong_shape_chain_averages_are_rejected(self):
        path = self.paper / "data_80k/t3_abs/means_t3_abs.npz"
        with np.load(path) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        original = arrays["ula"].copy()
        for bad in (np.full(2000, np.nan), original[:-1], original.reshape(1000, 2)):
            with self.subTest(shape=bad.shape):
                arrays["ula"] = bad
                np.savez_compressed(path, **arrays)
                with self.assertRaisesRegex(ValueError, "invalid chain-average"):
                    verify_paper(self.paper)

    def test_inconsistent_summary_is_rejected(self):
        self.edit_csv("data_80k/t3_abs/summary.csv", lambda rows: rows[1].__setitem__(-1, "1.0"))
        with self.assertRaisesRegex(ValueError, "inconsistent summary sd_of_chain_means"):
            verify_paper(self.paper)

    def test_source_hash_mismatch_is_rejected(self):
        self.edit_json("figures/manifest.json",
                       lambda x: x["sources"]["t3_abs"].update(sha256="0"*64))
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            verify_paper(self.paper)

    def test_wrong_panel_mapping_is_rejected(self):
        self.edit_json("figures/manifest.json",
                       lambda x: x["figures"]["01_ula_fixed_random"][0].update(method="uhmc_10"))
        with self.assertRaisesRegex(ValueError, "incorrect panel mapping"):
            verify_paper(self.paper)

    def test_trimmed_qq_axis_is_rejected(self):
        self.edit_json("figures/manifest.json",
                       lambda x: x["figures"]["03_hmc_nuts"][2].update(ylim=[.13, .17]))
        with self.assertRaisesRegex(ValueError, "inconsistent QQ ylim"):
            verify_paper(self.paper)

    def test_missing_artifact_returns_failed_cli_status(self):
        (self.paper / "figures/03_hmc_nuts.pdf").unlink()
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            self.assertEqual(main(["--paper-dir", str(self.paper)]), 1)
        self.assertIn("Missing/empty artifact", output.getvalue())


if __name__ == "__main__":
    unittest.main()
