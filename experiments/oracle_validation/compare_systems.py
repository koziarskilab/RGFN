#!/usr/bin/env python3
"""Cross-system comparison: does the neosubstrate-contact differential separate glues from decoys?

5HXB/CRBN  -> GSPT1 differential = (CRBN+GSPT1) - (CRBN)        [warhead-anchored docking]
6TD3/CR8   -> DDB1  differential = (CDK12+DDB1) - (CDK12)       [straight box docking]

Same idea, two systems. The headline number is the known-minus-decoy GAP on the differential:
a big gap means the proxy rewards productive neosubstrate contact (a real glue signal); a flat
gap means the proxy only reads E3-pocket binding (the CRBN ceiling we hit earlier).

Set CRBN_DIR and TD3_DIR (default: the repo result locations).
"""
import csv
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
CRBN_DIR = os.environ.get("CRBN_DIR", os.path.join(HERE, "docking_crbn"))
TD3_DIR = os.environ.get("TD3_DIR", os.path.join(HERE, "docking_6td3"))

SYSTEMS = [
    (
        "5HXB / CRBN-GSPT1",
        CRBN_DIR,
        "known_crbn_results.csv",
        "decoy_crbn_results.csv",
        "gspt1_dvina",
        "gspt1_dcnnaff",
    ),
    (
        "6TD3 / CDK12-DDB1",
        TD3_DIR,
        "known_results.csv",
        "decoy_cdk_results.csv",
        "ddb1_dvina",
        "ddb1_dcnnaff",
    ),
]


def load(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else None


def col(rows, c):
    out = []
    for r in rows:
        if r.get("status") != "ok":
            continue
        v = r.get(c, "")
        if v not in ("", None):
            try:
                out.append(float(v))
            except ValueError:
                pass
    return out


def med(xs):
    return st.median(xs) if xs else float("nan")


def frac(xs, pred):
    return sum(1 for x in xs if pred(x)) / len(xs) if xs else float("nan")


def block(name, d, kf, df, dvina, dcnn):
    known, decoy = load(os.path.join(d, kf)), load(os.path.join(d, df))
    print(f"\n{'='*66}\n{name}")
    if known is None or decoy is None:
        print(f"  results not present ({kf}: {known is not None}, {df}: {decoy is not None})")
        return

    def stats(rows):
        return {
            "ok": sum(1 for r in rows if r.get("status") == "ok"),
            "n": len(rows),
            "vina_t2": med(col(rows, "vina_t2")),
            "cnnaff_t2": med(col(rows, "cnnaff_t2")),
            "dvina": med(col(rows, dvina)),
            "dcnn": med(col(rows, dcnn)),
            "frac_strong": frac(col(rows, dvina), lambda x: x < -1.5),
            "frac_v10": frac(col(rows, "vina_t2"), lambda x: x < -10),
        }

    k, dd = stats(known), stats(decoy)
    print(f"  fit-rate: known {k['ok']}/{k['n']}  decoy {dd['ok']}/{dd['n']}")
    print(f"  {'metric':<26}{'known':>10}{'decoy':>10}{'gap':>10}")

    def line(lab, a, b, pct=False):
        f = (lambda x: f"{x*100:6.1f}%") if pct else (lambda x: f"{x:8.2f}")
        g = (a - b) * (100 if pct else 1)
        print(f"  {lab:<26}{f(a):>10}{f(b):>10}{g:>9.2f}{'%' if pct else ''}")

    line("median Tier2 Vina", k["vina_t2"], dd["vina_t2"])
    line("median Tier2 CNNaff", k["cnnaff_t2"], dd["cnnaff_t2"])
    line("median neosub dVina", k["dvina"], dd["dvina"])
    line("median neosub dCNNaff", k["dcnn"], dd["dcnn"])
    line("frac dVina < -1.5", k["frac_strong"], dd["frac_strong"], pct=True)
    line("frac Tier2 Vina < -10", k["frac_v10"], dd["frac_v10"], pct=True)
    return (k, dd)


def main():
    res = {}
    for name, d, kf, df, dvina, dcnn in SYSTEMS:
        res[name] = block(name, d, kf, df, dvina, dcnn)
    print(f"\n{'='*66}\nHEADLINE: known-minus-decoy gap on the neosubstrate differential")
    for name, _, _, _, _, _ in SYSTEMS:
        if res.get(name):
            k, dd = res[name]
            gap = k["dvina"] - dd["dvina"]
            strong_gap = (k["frac_strong"] - dd["frac_strong"]) * 100
            print(
                f"  {name:<22} median dVina gap {gap:+.2f} kcal/mol | "
                f"strong-bonus gap {strong_gap:+.0f} pts"
            )


if __name__ == "__main__":
    main()
