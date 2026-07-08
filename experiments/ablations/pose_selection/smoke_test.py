"""Off-Balam smoke test for the pose-selection ablation (entry 008).

Confirms the pipeline is SOUND without a GPU/gnina/Balam, so a future agent can
verify the code before spending a compute allocation. It exercises every part
that does NOT need gnina:

  - dock_allposes.load_all(): the molecule set loads with both classes present;
  - dock_allposes.embed(): RDKit 3D embedding produces a molblock;
  - dock_allposes._poses(): a fixture multi-pose SDF (with gnina-style property
    tags) groups by molecule and parses vina/cnnsc/cnnaff;
  - analyze_pose_selection.py: runs end-to-end on synthetic per-pose CSVs and
    emits a well-formed stats table with both selection rules.

What it does NOT test: the actual gnina docking call (Balam/GPU only). A
real run still needs Balam — this only proves the surrounding logic is intact.

Run:  python experiments/ablations/pose_selection/smoke_test.py
Exit: 0 = passed, non-zero = a check failed (so CI / an agent can gate on it).
"""

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_load_and_embed(da):
    mols = da.load_all()
    nk = sum(1 for m in mols if m[2] == "known")
    nd = sum(1 for m in mols if m[2] == "decoy")
    assert nk > 0 and nd > 0, f"load_all should return both classes (known={nk}, decoy={nd})"
    blk = da.embed(mols[0])[1]
    assert blk and "V2000" in blk, "embed should return an RDKit molblock"
    print(f"  [ok] load_all: {nk} known / {nd} decoy; embed produced a 3D molblock")


def test_poses_parsing(da):
    """Build a fixture multi-pose SDF with gnina-style tags and check _poses groups it."""
    base = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    from rdkit.Chem import AllChem

    AllChem.EmbedMolecule(base, randomSeed=1)
    with tempfile.TemporaryDirectory() as d:
        sdf = str(Path(d) / "fixture.sdf")
        w = Chem.SDWriter(sdf)
        for idx in ("1", "2"):
            for rank in range(3):  # 3 poses per molecule
                m = Chem.Mol(base)
                m.SetProp("_Name", idx)
                m.SetProp("minimizedAffinity", f"{-8.0 - rank:.3f}")
                m.SetProp("CNNscore", f"{0.5 + 0.1 * rank:.4f}")
                m.SetProp("CNNaffinity", f"{5.0 + rank:.4f}")
                w.write(m)
        w.close()
        groups = da._poses(sdf)
    assert set(groups) == {"1", "2"}, f"expected idx keys 1,2; got {set(groups)}"
    assert all(len(v) == 3 for v in groups.values()), "expected 3 poses per molecule"
    p = groups["1"][0]
    assert {"vina", "cnnsc", "cnnaff"} <= set(p) and isinstance(p["vina"], float)
    assert len(da.HEADER) == 12, f"per-pose CSV HEADER should have 12 columns, got {len(da.HEADER)}"
    print("  [ok] _poses: grouped 2 molecules x 3 poses; vina/cnnsc/cnnaff parsed")


def _synth_csv(path, set_label, n, glue, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for idx in range(n):
        for rank in range(1, 4):
            v2 = rng.normal(-9 if glue else -8, 1.0)
            v1 = v2 + rng.normal(2.0 if glue else 0.3, 0.5)  # glues gain more T2->T1
            rows.append(
                [
                    idx,
                    f"{set_label}{idx}",
                    set_label,
                    "CCO",
                    "ok",
                    3,
                    rank,
                    round(v2, 3),
                    round(rng.uniform(0, 1), 4),
                    round(rng.uniform(4, 7), 4),
                    round(v1, 3),
                    round(rng.uniform(4, 7), 4),
                ]
            )
    cols = [
        "idx",
        "id",
        "set",
        "smiles",
        "status",
        "n_poses",
        "pose_rank",
        "vina_t2",
        "cnnsc_t2",
        "cnnaff_t2",
        "vina_t1",
        "cnnaff_t1",
    ]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


def test_analysis_end_to_end():
    """Run analyze_pose_selection.py on synthetic per-pose CSVs; check the stats table."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _synth_csv(d / "known_allposes.csv", "known", 40, True, 0)
        _synth_csv(d / "decoy_allposes.csv", "decoy", 50, False, 1)
        env = {"DATA_DIR": str(d), "OUT_DIR": str(d)}
        r = subprocess.run(
            [sys.executable, str(HERE / "analyze_pose_selection.py")],
            capture_output=True,
            text=True,
            env={**_os_environ(), **env},
        )
        assert r.returncode == 0, f"analyze failed:\n{r.stderr}"
        stats = pd.read_csv(d / "pose_selection_stats.csv")
        assert set(stats["rule"]) == {"cnn", "vina"}, "stats must cover both selection rules"
        assert stats["auroc"].between(0, 1).all(), "AUROC must be in [0,1]"
        assert (d / "pose_selection_violins.png").exists(), "figure should be produced"
    print("  [ok] analyze_pose_selection: ran end-to-end; stats table + figure well-formed")


def _os_environ():
    import os

    return dict(os.environ)


def main():
    print("pose-selection ablation smoke test (no gnina/Balam):")
    da = _load("dock_allposes")
    test_load_and_embed(da)
    test_poses_parsing(da)
    test_analysis_end_to_end()
    print("SMOKE TEST PASSED — logic is sound; the docking call still needs Balam.")


if __name__ == "__main__":
    main()
