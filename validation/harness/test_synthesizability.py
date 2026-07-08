"""Local, dependency-free tests for the synthesizability evaluator's orchestration.

These exercise read -> dedup -> top_k -> search -> scatter-back -> aggregate WITHOUT
RDKit or AiZynthFinder installed, by monkeypatching the three external touch-points
(``_canonical``, ``sa_score``, ``run_aizynth``). So they run anywhere (the `aizynth`
env is only needed for a real run). Run:  python -m unittest validation.harness.test_synthesizability
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from validation.harness import synthesizability as S


def _write_dataset(d: Path, rows, manifest=None):
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "manifest.json", "w") as fh:
        json.dump(
            manifest
            or {
                "generator": "test",
                "system": "6td3",
                "oracle": "mock",
                "seed": 0,
                "score_higher_is_better": False,
            },
            fh,
        )
    cols = ["candidate_id", "smiles", "generator", "score", "has_route", "num_reactions"]
    with open(d / "candidates.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


class SynthesizabilityOrchestration(unittest.TestCase):
    def setUp(self):
        # Identity canonicalization + deterministic stub SA score, no RDKit needed.
        self._orig_canon = S._canonical
        self._orig_sa = S.sa_score
        self._orig_run = S.run_aizynth
        S._canonical = lambda smi: (smi or None)
        S.sa_score = lambda smi: (float(len(smi)) if smi else None)

    def tearDown(self):
        S._canonical = self._orig_canon
        S.sa_score = self._orig_sa
        S.run_aizynth = self._orig_run

    def _eval(self, tmp, **kw):
        return S.evaluate_dataset(
            Path(tmp) / "ds", out_dir=Path(tmp) / "out", config="unused", **kw
        )

    def test_dedup_and_success_rate(self):
        rows = [
            {
                "candidate_id": "a",
                "smiles": "CCO",
                "score": "-3.0",
                "has_route": "1",
                "num_reactions": "2",
            },
            {
                "candidate_id": "b",
                "smiles": "CCO",
                "score": "-2.0",  # dup of a
                "has_route": "1",
                "num_reactions": "2",
            },
            {
                "candidate_id": "c",
                "smiles": "c1ccccc1",
                "score": "-1.0",
                "has_route": "0",
                "num_reactions": "",
            },
            {
                "candidate_id": "d",
                "smiles": "",
                "score": "",  # invalid
                "has_route": "0",
                "num_reactions": "",
            },
        ]
        # AiZynth should be called once per UNIQUE valid SMILES (CCO, benzene).
        captured = {}

        def fake_run(smiles_list, **kw):
            captured["smiles"] = list(smiles_list)
            verdict = {"CCO": 1, "c1ccccc1": 0}
            return [
                {
                    "solved": verdict[s],
                    "n_steps": (3 if verdict[s] else None),
                    "n_solved_routes": (1 if verdict[s] else 0),
                    "top_score": 0.9,
                    "search_time": 0.1,
                    "error": None,
                }
                for s in smiles_list
            ]

        S.run_aizynth = fake_run
        with tempfile.TemporaryDirectory() as tmp:
            _write_dataset(Path(tmp) / "ds", rows)
            summ = self._eval(tmp)

            # searched 2 unique valid molecules, not 3 valid rows
            self.assertEqual(sorted(captured["smiles"]), ["CCO", "c1ccccc1"])
            self.assertEqual(summ["n_candidates"], 4)
            self.assertEqual(summ["n_valid"], 3)
            self.assertEqual(summ["n_evaluated"], 2)
            self.assertEqual(summ["n_solved"], 1)  # only CCO solved
            self.assertAlmostEqual(summ["aizynth_success_rate"], 0.5)
            self.assertAlmostEqual(summ["steps_mean"], 3.0)
            self.assertAlmostEqual(summ["self_reported_route_rate"], 0.5)  # 2 of 4

            # per-molecule CSV: the duplicate row b inherits a's solved verdict
            with open(Path(tmp) / "out" / "synthesizability.csv") as fh:
                per = {r["candidate_id"]: r for r in csv.DictReader(fh)}
            self.assertEqual(per["a"]["solved"], "1")
            self.assertEqual(per["b"]["solved"], "1")  # scattered back onto dup
            self.assertEqual(per["c"]["solved"], "0")
            self.assertEqual(per["d"]["valid"], "False")
            self.assertEqual(per["d"]["evaluated"], "False")

    def test_top_k_selects_best_by_score(self):
        rows = [
            {
                "candidate_id": str(i),
                "smiles": f"C{'C'*i}O",
                "score": str(-i),
                "has_route": "0",
                "num_reactions": "",
            }
            for i in range(5)
        ]
        seen = {}

        def fake_run(smiles_list, **kw):
            seen["n"] = len(smiles_list)
            return [
                {
                    "solved": 1,
                    "n_steps": 1,
                    "n_solved_routes": 1,
                    "top_score": 1.0,
                    "search_time": 0.0,
                    "error": None,
                }
                for _ in smiles_list
            ]

        S.run_aizynth = fake_run
        with tempfile.TemporaryDirectory() as tmp:
            _write_dataset(Path(tmp) / "ds", rows)
            # lower-is-better (score_higher_is_better=False) -> best are scores -4,-3
            summ = self._eval(tmp, top_k=2)
            self.assertEqual(seen["n"], 2)
            self.assertEqual(summ["n_evaluated"], 2)

    def test_output_files_written(self):
        S.run_aizynth = lambda smiles_list, **kw: [
            {
                "solved": 0,
                "n_steps": None,
                "n_solved_routes": 0,
                "top_score": None,
                "search_time": 0.0,
                "error": None,
            }
            for _ in smiles_list
        ]
        with tempfile.TemporaryDirectory() as tmp:
            _write_dataset(
                Path(tmp) / "ds",
                [
                    {
                        "candidate_id": "x",
                        "smiles": "CCO",
                        "score": "-1",
                        "has_route": "0",
                        "num_reactions": "",
                    }
                ],
            )
            self._eval(tmp)
            out = Path(tmp) / "out"
            self.assertTrue((out / "synthesizability.csv").exists())
            self.assertTrue((out / "synthesizability_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
