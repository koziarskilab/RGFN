# Matched four-way fixed-reward 6td3 benchmark (RGFN / FragGFN / RxnFlow / SCENT)

| Generator | Synth? | n_uniq | n_eval | AiZynth success | steps | SA | self-route | MW | QED |
|---|---|---|---|---|---|---|---|---|---|
| rgfn | yes | 100 | 100 | 0.160 | 5 | 3.776 | 1.000 | 679.116 | 0.114 |
| fraggfn | no | 100 | 100 | 0.000 |  | 4.671 | 0.000 | 677.236 | 0.191 |
| rxnflow | yes | 100 | 100 | 0.250 | 4.600 | 3.721 | 1.000 | 536.884 | 0.188 |
| scent | yes | 100 | 100 | 0.210 | 4.476 | 3.528 | 1.000 | 629.709 | 0.135 |

*AiZynth success = fraction of unique valid molecules with a full retrosynthetic route to in-stock precursors (the headline synthesizability metric). self-route = the generator's by-construction route claim (RGFN/RxnFlow/SCENT = 1.0; FragGFN = 0).*
