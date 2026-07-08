# Matched four-way fixed-reward sEH benchmark (RGFN / FragGFN / RxnFlow / SCENT)

| Generator | Synth? | n_uniq | n_eval | AiZynth success | steps | SA | self-route | MW | QED |
|---|---|---|---|---|---|---|---|---|---|
| rgfn | yes | 100 | 100 | 0.520 | 3.692 | 3.215 | 1.000 | 535.372 | 0.255 |
| fraggfn | no | 100 | 100 | 0.020 | 3.500 | 4.321 | 0.000 | 689.817 | 0.168 |
| rxnflow | yes | 100 | 100 | 0.330 | 4.424 | 4.652 | 1.000 | 718.044 | 0.201 |
| scent | yes | 100 | 100 | 0.710 | 4.014 | 3.129 | 1.000 | 514.111 | 0.288 |

*AiZynth success = fraction of unique valid molecules with a full retrosynthetic route to in-stock precursors (the headline synthesizability metric). self-route = the generator's by-construction route claim (RGFN/RxnFlow/SCENT = 1.0; FragGFN = 0).*
