import json
from pathlib import Path
import unittest

import numpy as np
import torch

from app import load_bundle, render
from sequence import CACHE, CHANNELS, CLASSES, HELD, ROOT, STEPS, SignLSTM, load_archive, resample


class ModelTests(unittest.TestCase):
    def test_preprocess_shape_and_roll_wrap(self):
        fake = "0,1,0,0,-1,-1,0,1,0,1\n1,2,1,1,-1,-1,1,0,1,0\n"
        data = resample(fake)
        self.assertEqual(data.shape, (STEPS,CHANNELS))
        self.assertTrue(np.allclose(data[:,3], 0, atol=2e-7))
        self.assertTrue(np.allclose(data[:,4], 1))
        self.assertAlmostEqual(float(data[-1,0]), 1)

    def test_five_distinct_group_tests(self):
        data = json.loads((ROOT/"report.json").read_text(encoding="utf-8"))
        self.assertEqual({f["held_out"] for f in data["folds"]}, set("ABCDE"))
        self.assertTrue(all(f["held_out"] != f["validation"] for f in data["folds"]))
        self.assertEqual(sum(f["test"] for f in data["folds"]), data["total"])
        self.assertEqual(np.asarray(data["confusion"]).sum(), data["total"])

    def test_bundled_weights_load(self):
        torch.set_num_threads(2)
        ckpt = torch.load(ROOT/"model.pt", weights_only=True, map_location="cpu")
        net = SignLSTM()
        net.load_state_dict(ckpt["state"])
        with torch.inference_mode():
            out = net.eval()(torch.zeros(2,STEPS,CHANNELS))
        self.assertEqual(out.shape, (2,len(CLASSES)))
        self.assertTrue(torch.isfinite(out).all())

    @unittest.skipUnless(CACHE.is_file(), "Run app.py once or download the official data into .cache first.")
    def test_archive_and_ui(self):
        x, y, group, paths = load_archive()
        self.assertEqual(len(set(paths)), len(x))
        self.assertEqual(x.shape,(700,STEPS,CHANNELS))
        self.assertEqual(set(y), set(range(len(CLASSES))))
        self.assertEqual(np.count_nonzero(group == HELD),180)
        demo = load_bundle()
        self.assertTrue(np.allclose(demo[1].sum(1),1))
        self.assertIn("Ground truth",render(demo,0))
        self.assertNotEqual(render(demo,0),render(demo,1))
        self.assertIn("&lt;", render(demo,0,"<bad>"))


if __name__ == "__main__":
    unittest.main()
