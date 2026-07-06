# Same-library four-way fixed-reward sEH benchmark on glue_standard_v1 (RGFN / FragGFN / RxnFlow / SCENT)

| Generator | Synth? | n_uniq | n_eval | AiZynth success | steps | SA | self-route | MW | QED |
|---|---|---|---|---|---|---|---|---|---|
| rgfn | yes | 100 | 100 | 0.660 | 4.076 | 3.362 | 1.000 | 553.195 | 0.238 |
| fraggfn | no | 100 | 100 | 0.020 | 3.500 | 4.321 | 0.000 | 689.817 | 0.168 |
| rxnflow | yes | 100 | 100 | 0.290 | 4.069 | 3.455 | 1.000 | 412.642 | 0.621 |
| scent | yes | 100 | 100 | 0.710 | 4.070 | 3.129 | 1.000 | 514.111 | 0.288 |

*AiZynth success = fraction of unique valid molecules with a full retrosynthetic route to in-stock precursors (the headline synthesizability metric). self-route = the generator's by-construction route claim (RGFN/RxnFlow/SCENT = 1.0; FragGFN = 0).*
