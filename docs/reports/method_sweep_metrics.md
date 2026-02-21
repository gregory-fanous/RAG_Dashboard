# Backend Method Sweep (Real Mode)

| Dataset | Method | Backend | Quality | Hallucination | Precision@k | Recall@k | MRR | Latency (ms) | Avg Cost ($) | Total Cost ($) | Run ID |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| finder | baseline_fixed | in_memory | 0.7350 | 0.4333 | 0.2000 | 1.0000 | 1.0000 | 17858.6 | 0.02907 | 0.29066 | 20260213T232054Z |
| finder | graphrag_plus | in_memory | 0.5320 | 0.4200 | 0.1000 | 0.7000 | 0.7000 | 22728.2 | 0.03998 | 0.39983 | 20260213T232442Z |
| finder | logicrag_guarded | in_memory | 0.2867 | 0.8000 | 0.0667 | 0.4000 | 0.4000 | 25540.9 | 0.04812 | 0.48119 | 20260213T232915Z |
| open_ragbench | baseline_fixed | in_memory | 0.6925 | 0.2000 | 0.1800 | 0.9000 | 0.8500 | 10112.8 | 0.04075 | 0.40746 | 20260214T070755Z |
| open_ragbench | self_guarded | in_memory | 0.5463 | 0.3250 | 0.1000 | 0.7000 | 0.7000 | 23705.0 | 0.07792 | 0.77922 | 20260214T071736Z |
| open_ragbench | hybrid_dense | in_memory | 0.5200 | 0.1000 | 0.1000 | 0.6000 | 0.6000 | 16788.7 | 0.05171 | 0.51714 | 20260214T071043Z |
| open_ragbench | graphrag_plus | in_memory | 0.5014 | 0.2000 | 0.0857 | 0.6000 | 0.6000 | 17594.0 | 0.06700 | 0.67003 | 20260214T071339Z |
| open_ragbench | logicrag_guarded | in_memory | 0.4883 | 0.6000 | 0.1167 | 0.7000 | 0.6167 | 19061.1 | 0.04040 | 0.40401 | 20260214T072649Z |
| public_eval_set | baseline_fixed | in_memory | 0.8150 | 0.0000 | 0.2600 | 1.0000 | 1.0000 | 8777.6 | 0.00949 | 0.09491 | 20260213T072813Z |
| public_eval_set | logicrag_guarded | in_memory | 0.7442 | 0.2556 | 0.2000 | 0.9500 | 1.0000 | 11108.4 | 0.01364 | 0.13641 | 20260213T230636Z |
| ragcare_qa | self_guarded | in_memory | 0.7297 | 0.3733 | 0.1429 | 1.0000 | 1.0000 | 23015.2 | 0.03485 | 0.34848 | 20260213T075519Z |
| ragcare_qa | logicrag_guarded | in_memory | 0.5302 | 0.4600 | 0.1167 | 0.7000 | 0.7000 | 16463.3 | 0.02613 | 0.26129 | 20260213T230920Z |
| retrievalqa | graphrag_plus | in_memory | 0.6527 | 0.0000 | 0.3000 | 0.7875 | 0.6083 | 15823.0 | 0.03414 | 0.34138 | 20260214T070305Z |
| retrievalqa | self_guarded | in_memory | 0.6377 | 0.1000 | 0.3000 | 0.7875 | 0.6083 | 18854.2 | 0.03891 | 0.38906 | 20260214T070613Z |
| retrievalqa | hybrid_dense | in_memory | 0.6177 | 0.0000 | 0.3167 | 0.7208 | 0.5450 | 12786.6 | 0.03156 | 0.31557 | 20260214T070026Z |
| retrievalqa | baseline_fixed | in_memory | 0.5933 | 0.0500 | 0.3000 | 0.6250 | 0.6283 | 5471.6 | 0.01777 | 0.17769 | 20260214T065818Z |
| retrievalqa | logicrag_guarded | in_memory | 0.3192 | 0.6000 | 0.1333 | 0.4167 | 0.3200 | 13208.2 | 0.01518 | 0.15184 | 20260214T072338Z |
