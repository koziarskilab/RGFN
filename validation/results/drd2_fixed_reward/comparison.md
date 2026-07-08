# Matched four-way fixed-reward drd2 benchmark (RGFN / FragGFN / RxnFlow / SCENT)

| Generator | Synth? | n_uniq | n_eval | AiZynth success | steps | SA | self-route | MW | QED |
|---|---|---|---|---|---|---|---|---|---|
| rgfn |  |  |  |  |  |  |  |  |  |
| fraggfn | no | 100 | 100 | 0.000 |  | 4.667 | 0.000 | 680.176 | 0.193 |
| rxnflow | yes | 100 | 100 | 0.240 | 4.125 | 3.482 | 1.000 | 492.359 | 0.376 |
| scent | yes | 100 | 100 | 0.280 | 3.821 | 3.294 | 1.000 | 508.797 | 0.347 |

*AiZynth success = fraction of unique valid molecules with a full retrosynthetic route to in-stock precursors (the headline synthesizability metric). self-route = the generator's by-construction route claim (RGFN/RxnFlow/SCENT = 1.0; FragGFN = 0).*


> **Blank row(s) pending:** rgfn — run still training when this table was built. Re-run this system's AiZynth aggregation to fill in (see `Logs/021` ▶ PICK UP HERE).
