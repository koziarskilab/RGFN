# Matched four-way fixed-reward clpp benchmark (RGFN / FragGFN / RxnFlow / SCENT)

| Generator | Synth? | n_uniq | n_eval | AiZynth success | steps | SA | self-route | MW | QED |
|---|---|---|---|---|---|---|---|---|---|
| rgfn | yes | 100 | 100 | 0.200 | 3.900 | 3.723 | 1.000 | 608.216 | 0.191 |
| fraggfn | no | 100 | 100 | 0.000 |  | 4.766 | 0.000 | 564.408 | 0.285 |
| rxnflow | yes | 100 | 100 | 0.610 | 2.557 | 3.026 | 1.000 | 379.439 | 0.448 |
| scent | yes | 100 | 100 | 0.250 | 3.640 | 3.700 | 1.000 | 584.343 | 0.233 |

*AiZynth success = fraction of unique valid molecules with a full retrosynthetic route to in-stock precursors (the headline synthesizability metric). self-route = the generator's by-construction route claim (RGFN/RxnFlow/SCENT = 1.0; FragGFN = 0).*
