# Matched four-way fixed-reward DRD2 benchmark (RGFN / FragGFN / RxnFlow / SCENT)

| Generator | Synth? | n_uniq | n_eval | AiZynth success | steps | SA | self-route | MW | QED |
|---|---|---|---|---|---|---|---|---|---|
| rgfn | yes | 100 | 100 | 0.470 | 3.957 | 3.098 | 1.000 | 486.334 | 0.342 |
| fraggfn | no | 100 | 100 | 0.000 |  | 4.667 | 0.000 | 680.176 | 0.193 |
| rxnflow | yes | 100 | 100 | 0.300 | 4.133 | 3.326 | 1.000 | 533.625 | 0.287 |
| scent | yes | 100 | 100 | 0.250 | 3.680 | 3.294 | 1.000 | 508.797 | 0.347 |

*AiZynth success = fraction of unique valid molecules with a full retrosynthetic route to in-stock precursors (the headline synthesizability metric). self-route = the generator's by-construction route claim (RGFN/RxnFlow/SCENT = 1.0; FragGFN = 0).*
