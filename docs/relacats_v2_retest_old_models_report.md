# RelaCaTS-v2 old-model retest

This report was produced by the CPU aggregation stage after fresh GPU response and confidence generation.
No model merge, checkpoint write, or training was performed.

## Protocol

- Candidate responses per question: `32`
- Dynamic target and hard cap: `16`
- Calibration holdout: `0.200` of questions, SHA-256 question-id split, seed `42`
- Thresholds are selected on validation only and reloaded from `thresholds/` for held-out test.
- Invalid answers remain in the strict question/sample denominators.
- Legacy Excel reference: not found/supplied; comparisons use the existing machine-readable `table2_results.json/csv` only.

## Test metrics

| Model | Dataset | Method | Accuracy | Actual avg samples | Valid samples | Invalid rate |
|---|---|---|---:|---:|---:|---:|
| qwen2_5_7b_instruct_cats | object_counting | SC | 81.283% | 16.000 | 2992 | 0.000% |
| qwen2_5_7b_instruct_cats | object_counting | CISC | 82.353% | 16.000 | 2992 | 0.000% |
| qwen2_5_7b_instruct_cats | object_counting | Self-Certainty | 75.936% | 16.000 | 2992 | 0.000% |
| qwen2_5_7b_instruct_cats | object_counting | RelaCaTS-SC | 82.353% | 16.000 | 2992 | 0.000% |
| qwen2_5_7b_instruct_cats | object_counting | Best-of-N | 75.936% | 16.000 | 2992 | 0.000% |
| qwen2_5_7b_instruct_cats | object_counting | RelaCaTS-ES | 82.353% | 15.433 | 2886 | 0.000% |
| qwen2_5_7b_instruct_cats | object_counting | ASC | 81.283% | 5.979 | 1118 | 0.000% |
| qwen2_5_7b_instruct_cats | object_counting | RelaCaTS-ASC | 81.818% | 6.150 | 1150 | 0.000% |
| qwen2_5_7b_instruct_cats | object_counting | ESC | 81.283% | 16.000 | 2992 | 0.000% |
| qwen2_5_7b_instruct_cats | object_counting | RASC | 82.353% | 15.840 | 2962 | 0.000% |
| qwen2_5_7b_instruct_cats | math_qa | SC | 87.108% | 16.000 | 37850 | 0.978% |
| qwen2_5_7b_instruct_cats | math_qa | CISC | 87.652% | 16.000 | 37850 | 0.978% |
| qwen2_5_7b_instruct_cats | math_qa | Self-Certainty | 87.735% | 16.000 | 37850 | 0.978% |
| qwen2_5_7b_instruct_cats | math_qa | RelaCaTS-SC | 88.866% | 16.000 | 37850 | 0.978% |
| qwen2_5_7b_instruct_cats | math_qa | Best-of-N | 87.735% | 16.000 | 37850 | 0.978% |
| qwen2_5_7b_instruct_cats | math_qa | RelaCaTS-ES | 88.866% | 15.995 | 37838 | 0.979% |
| qwen2_5_7b_instruct_cats | math_qa | ASC | 87.149% | 4.435 | 10407 | 1.774% |
| qwen2_5_7b_instruct_cats | math_qa | RelaCaTS-ASC | 88.196% | 4.435 | 10407 | 1.774% |
| qwen2_5_7b_instruct_cats | math_qa | ESC | 87.108% | 16.000 | 37850 | 0.978% |
| qwen2_5_7b_instruct_cats | math_qa | RASC | 88.866% | 16.000 | 37850 | 0.978% |
| qwen2_5_7b_instruct_cats | arc_challenge | SC | 91.983% | 16.000 | 14730 | 0.257% |
| qwen2_5_7b_instruct_cats | arc_challenge | CISC | 91.983% | 16.000 | 14730 | 0.257% |
| qwen2_5_7b_instruct_cats | arc_challenge | Self-Certainty | 89.924% | 16.000 | 14730 | 0.257% |
| qwen2_5_7b_instruct_cats | arc_challenge | RelaCaTS-SC | 91.224% | 16.000 | 14730 | 0.257% |
| qwen2_5_7b_instruct_cats | arc_challenge | Best-of-N | 89.924% | 16.000 | 14730 | 0.257% |
| qwen2_5_7b_instruct_cats | arc_challenge | RelaCaTS-ES | 91.224% | 16.000 | 14730 | 0.257% |
| qwen2_5_7b_instruct_cats | arc_challenge | ASC | 91.441% | 3.035 | 2790 | 0.393% |
| qwen2_5_7b_instruct_cats | arc_challenge | RelaCaTS-ASC | 91.224% | 3.035 | 2790 | 0.393% |
| qwen2_5_7b_instruct_cats | arc_challenge | ESC | 91.983% | 16.000 | 14730 | 0.257% |
| qwen2_5_7b_instruct_cats | arc_challenge | RASC | 91.224% | 16.000 | 14730 | 0.257% |
| qwen2_5_7b_instruct_relacats_v1 | object_counting | SC | 79.679% | 16.000 | 2992 | 0.000% |
| qwen2_5_7b_instruct_relacats_v1 | object_counting | CISC | 79.679% | 16.000 | 2992 | 0.000% |
| qwen2_5_7b_instruct_relacats_v1 | object_counting | Self-Certainty | 76.471% | 16.000 | 2992 | 0.000% |
| qwen2_5_7b_instruct_relacats_v1 | object_counting | RelaCaTS-SC | 79.679% | 16.000 | 2992 | 0.000% |
| qwen2_5_7b_instruct_relacats_v1 | object_counting | Best-of-N | 76.471% | 16.000 | 2992 | 0.000% |
| qwen2_5_7b_instruct_relacats_v1 | object_counting | RelaCaTS-ES | 79.679% | 15.930 | 2979 | 0.000% |
| qwen2_5_7b_instruct_relacats_v1 | object_counting | ASC | 77.540% | 5.754 | 1076 | 0.000% |
| qwen2_5_7b_instruct_relacats_v1 | object_counting | RelaCaTS-ASC | 77.540% | 6.016 | 1125 | 0.000% |
| qwen2_5_7b_instruct_relacats_v1 | object_counting | ESC | 79.679% | 16.000 | 2992 | 0.000% |
| qwen2_5_7b_instruct_relacats_v1 | object_counting | RASC | 79.679% | 15.893 | 2972 | 0.000% |
| qwen2_5_7b_instruct_relacats_v1 | math_qa | SC | 87.987% | 16.000 | 37985 | 0.625% |
| qwen2_5_7b_instruct_relacats_v1 | math_qa | CISC | 88.321% | 16.000 | 37985 | 0.625% |
| qwen2_5_7b_instruct_relacats_v1 | math_qa | Self-Certainty | 86.982% | 16.000 | 37985 | 0.625% |
| qwen2_5_7b_instruct_relacats_v1 | math_qa | RelaCaTS-SC | 88.740% | 16.000 | 37985 | 0.625% |
| qwen2_5_7b_instruct_relacats_v1 | math_qa | Best-of-N | 86.982% | 16.000 | 37985 | 0.625% |
| qwen2_5_7b_instruct_relacats_v1 | math_qa | RelaCaTS-ES | 88.740% | 16.000 | 37985 | 0.625% |
| qwen2_5_7b_instruct_relacats_v1 | math_qa | ASC | 87.526% | 3.966 | 9369 | 1.108% |
| qwen2_5_7b_instruct_relacats_v1 | math_qa | RelaCaTS-ASC | 87.903% | 3.965 | 9367 | 1.109% |
| qwen2_5_7b_instruct_relacats_v1 | math_qa | ESC | 87.987% | 16.000 | 37985 | 0.625% |
| qwen2_5_7b_instruct_relacats_v1 | math_qa | RASC | 88.740% | 16.000 | 37985 | 0.625% |
| qwen2_5_7b_instruct_relacats_v1 | arc_challenge | SC | 91.874% | 16.000 | 14731 | 0.251% |
| qwen2_5_7b_instruct_relacats_v1 | arc_challenge | CISC | 91.766% | 16.000 | 14731 | 0.251% |
| qwen2_5_7b_instruct_relacats_v1 | arc_challenge | Self-Certainty | 90.249% | 16.000 | 14731 | 0.251% |
| qwen2_5_7b_instruct_relacats_v1 | arc_challenge | RelaCaTS-SC | 91.658% | 16.000 | 14731 | 0.251% |
| qwen2_5_7b_instruct_relacats_v1 | arc_challenge | Best-of-N | 90.249% | 16.000 | 14731 | 0.251% |
| qwen2_5_7b_instruct_relacats_v1 | arc_challenge | RelaCaTS-ES | 91.658% | 16.000 | 14731 | 0.251% |
| qwen2_5_7b_instruct_relacats_v1 | arc_challenge | ASC | 92.416% | 2.728 | 2506 | 0.477% |
| qwen2_5_7b_instruct_relacats_v1 | arc_challenge | RelaCaTS-ASC | 92.199% | 2.715 | 2494 | 0.479% |
| qwen2_5_7b_instruct_relacats_v1 | arc_challenge | ESC | 91.874% | 16.000 | 14731 | 0.251% |
| qwen2_5_7b_instruct_relacats_v1 | arc_challenge | RASC | 91.658% | 16.000 | 14731 | 0.251% |
| llama3_1_8b_instruct_cats | object_counting | SC | 77.540% | 16.000 | 2947 | 1.504% |
| llama3_1_8b_instruct_cats | object_counting | CISC | 76.471% | 16.000 | 2947 | 1.504% |
| llama3_1_8b_instruct_cats | object_counting | Self-Certainty | 78.075% | 16.000 | 2947 | 1.504% |
| llama3_1_8b_instruct_cats | object_counting | RelaCaTS-SC | 78.075% | 16.000 | 2947 | 1.504% |
| llama3_1_8b_instruct_cats | object_counting | Best-of-N | 78.075% | 16.000 | 2947 | 1.504% |
| llama3_1_8b_instruct_cats | object_counting | RelaCaTS-ES | 78.075% | 15.856 | 2920 | 1.518% |
| llama3_1_8b_instruct_cats | object_counting | ASC | 78.075% | 7.551 | 1374 | 2.691% |
| llama3_1_8b_instruct_cats | object_counting | RelaCaTS-ASC | 77.540% | 7.551 | 1374 | 2.691% |
| llama3_1_8b_instruct_cats | object_counting | ESC | 77.540% | 16.000 | 2947 | 1.504% |
| llama3_1_8b_instruct_cats | object_counting | RASC | 78.075% | 15.658 | 2884 | 1.503% |
| llama3_1_8b_instruct_cats | math_qa | SC | 83.005% | 16.000 | 29413 | 23.051% |
| llama3_1_8b_instruct_cats | math_qa | CISC | 84.261% | 16.000 | 29413 | 23.051% |
| llama3_1_8b_instruct_cats | math_qa | Self-Certainty | 81.457% | 16.000 | 29413 | 23.051% |
| llama3_1_8b_instruct_cats | math_qa | RelaCaTS-SC | 84.847% | 16.000 | 29413 | 23.051% |
| llama3_1_8b_instruct_cats | math_qa | Best-of-N | 81.457% | 16.000 | 29413 | 23.051% |
| llama3_1_8b_instruct_cats | math_qa | RelaCaTS-ES | 84.847% | 16.000 | 29413 | 23.051% |
| llama3_1_8b_instruct_cats | math_qa | ASC | 82.419% | 7.196 | 11570 | 32.697% |
| llama3_1_8b_instruct_cats | math_qa | RelaCaTS-ASC | 83.508% | 7.196 | 11570 | 32.697% |
| llama3_1_8b_instruct_cats | math_qa | ESC | 83.005% | 16.000 | 29413 | 23.051% |
| llama3_1_8b_instruct_cats | math_qa | RASC | 84.847% | 16.000 | 29413 | 23.051% |
| llama3_1_8b_instruct_cats | arc_challenge | SC | 89.274% | 16.000 | 14469 | 2.025% |
| llama3_1_8b_instruct_cats | arc_challenge | CISC | 89.707% | 16.000 | 14469 | 2.025% |
| llama3_1_8b_instruct_cats | arc_challenge | Self-Certainty | 87.432% | 16.000 | 14469 | 2.025% |
| llama3_1_8b_instruct_cats | arc_challenge | RelaCaTS-SC | 89.166% | 16.000 | 14469 | 2.025% |
| llama3_1_8b_instruct_cats | arc_challenge | Best-of-N | 87.432% | 16.000 | 14469 | 2.025% |
| llama3_1_8b_instruct_cats | arc_challenge | RelaCaTS-ES | 89.166% | 16.000 | 14469 | 2.025% |
| llama3_1_8b_instruct_cats | arc_challenge | ASC | 88.624% | 4.137 | 3675 | 3.745% |
| llama3_1_8b_instruct_cats | arc_challenge | RelaCaTS-ASC | 88.624% | 4.137 | 3675 | 3.745% |
| llama3_1_8b_instruct_cats | arc_challenge | ESC | 89.274% | 16.000 | 14469 | 2.025% |
| llama3_1_8b_instruct_cats | arc_challenge | RASC | 89.166% | 16.000 | 14469 | 2.025% |
| llama3_1_8b_instruct_relacats_v1 | object_counting | SC | 56.684% | 16.000 | 2947 | 1.504% |
| llama3_1_8b_instruct_relacats_v1 | object_counting | CISC | 57.754% | 16.000 | 2947 | 1.504% |
| llama3_1_8b_instruct_relacats_v1 | object_counting | Self-Certainty | 57.754% | 16.000 | 2947 | 1.504% |
| llama3_1_8b_instruct_relacats_v1 | object_counting | RelaCaTS-SC | 58.289% | 16.000 | 2947 | 1.504% |
| llama3_1_8b_instruct_relacats_v1 | object_counting | Best-of-N | 57.754% | 16.000 | 2947 | 1.504% |
| llama3_1_8b_instruct_relacats_v1 | object_counting | RelaCaTS-ES | 58.289% | 16.000 | 2947 | 1.504% |
| llama3_1_8b_instruct_relacats_v1 | object_counting | ASC | 56.150% | 7.706 | 1409 | 2.221% |
| llama3_1_8b_instruct_relacats_v1 | object_counting | RelaCaTS-ASC | 57.219% | 7.706 | 1409 | 2.221% |
| llama3_1_8b_instruct_relacats_v1 | object_counting | ESC | 56.684% | 16.000 | 2947 | 1.504% |
| llama3_1_8b_instruct_relacats_v1 | object_counting | RASC | 58.289% | 16.000 | 2947 | 1.504% |
| llama3_1_8b_instruct_relacats_v1 | math_qa | SC | 81.289% | 16.000 | 28592 | 25.199% |
| llama3_1_8b_instruct_relacats_v1 | math_qa | CISC | 82.252% | 16.000 | 28592 | 25.199% |
| llama3_1_8b_instruct_relacats_v1 | math_qa | Self-Certainty | 78.694% | 16.000 | 28592 | 25.199% |
| llama3_1_8b_instruct_relacats_v1 | math_qa | RelaCaTS-SC | 82.964% | 16.000 | 28592 | 25.199% |
| llama3_1_8b_instruct_relacats_v1 | math_qa | Best-of-N | 78.694% | 16.000 | 28592 | 25.199% |
| llama3_1_8b_instruct_relacats_v1 | math_qa | RelaCaTS-ES | 82.964% | 16.000 | 28592 | 25.199% |
| llama3_1_8b_instruct_relacats_v1 | math_qa | ASC | 80.075% | 7.566 | 11624 | 35.690% |
| llama3_1_8b_instruct_relacats_v1 | math_qa | RelaCaTS-ASC | 81.457% | 7.560 | 11612 | 35.703% |
| llama3_1_8b_instruct_relacats_v1 | math_qa | ESC | 81.289% | 16.000 | 28592 | 25.199% |
| llama3_1_8b_instruct_relacats_v1 | math_qa | RASC | 82.964% | 16.000 | 28592 | 25.199% |
| llama3_1_8b_instruct_relacats_v1 | arc_challenge | SC | 87.107% | 16.000 | 14567 | 1.361% |
| llama3_1_8b_instruct_relacats_v1 | arc_challenge | CISC | 86.566% | 16.000 | 14567 | 1.361% |
| llama3_1_8b_instruct_relacats_v1 | arc_challenge | Self-Certainty | 83.857% | 16.000 | 14567 | 1.361% |
| llama3_1_8b_instruct_relacats_v1 | arc_challenge | RelaCaTS-SC | 86.566% | 16.000 | 14567 | 1.361% |
| llama3_1_8b_instruct_relacats_v1 | arc_challenge | Best-of-N | 83.857% | 16.000 | 14567 | 1.361% |
| llama3_1_8b_instruct_relacats_v1 | arc_challenge | RelaCaTS-ES | 86.566% | 16.000 | 14567 | 1.361% |
| llama3_1_8b_instruct_relacats_v1 | arc_challenge | ASC | 86.566% | 3.607 | 3243 | 2.583% |
| llama3_1_8b_instruct_relacats_v1 | arc_challenge | RelaCaTS-ASC | 86.782% | 3.607 | 3243 | 2.583% |
| llama3_1_8b_instruct_relacats_v1 | arc_challenge | ESC | 87.107% | 16.000 | 14567 | 1.361% |
| llama3_1_8b_instruct_relacats_v1 | arc_challenge | RASC | 86.566% | 16.000 | 14567 | 1.361% |
| deepseek_r1_distill_qwen_1_5b_cats | object_counting | SC | 73.797% | 16.000 | 2682 | 10.361% |
| deepseek_r1_distill_qwen_1_5b_cats | object_counting | CISC | 72.727% | 16.000 | 2682 | 10.361% |
| deepseek_r1_distill_qwen_1_5b_cats | object_counting | Self-Certainty | 55.615% | 16.000 | 2682 | 10.361% |
| deepseek_r1_distill_qwen_1_5b_cats | object_counting | RelaCaTS-SC | 67.914% | 16.000 | 2682 | 10.361% |
| deepseek_r1_distill_qwen_1_5b_cats | object_counting | Best-of-N | 55.615% | 16.000 | 2682 | 10.361% |
| deepseek_r1_distill_qwen_1_5b_cats | object_counting | RelaCaTS-ES | 67.914% | 16.000 | 2682 | 10.361% |
| deepseek_r1_distill_qwen_1_5b_cats | object_counting | ASC | 72.727% | 8.872 | 1463 | 11.814% |
| deepseek_r1_distill_qwen_1_5b_cats | object_counting | RelaCaTS-ASC | 64.706% | 6.743 | 1119 | 11.261% |
| deepseek_r1_distill_qwen_1_5b_cats | object_counting | ESC | 73.797% | 16.000 | 2682 | 10.361% |
| deepseek_r1_distill_qwen_1_5b_cats | object_counting | RASC | 67.914% | 16.000 | 2682 | 10.361% |
| deepseek_r1_distill_qwen_1_5b_cats | math_qa | SC | 86.061% | 16.000 | 27680 | 27.585% |
| deepseek_r1_distill_qwen_1_5b_cats | math_qa | CISC | 85.977% | 16.000 | 27680 | 27.585% |
| deepseek_r1_distill_qwen_1_5b_cats | math_qa | Self-Certainty | 81.750% | 16.000 | 27680 | 27.585% |
| deepseek_r1_distill_qwen_1_5b_cats | math_qa | RelaCaTS-SC | 84.638% | 16.000 | 27680 | 27.585% |
| deepseek_r1_distill_qwen_1_5b_cats | math_qa | Best-of-N | 81.750% | 16.000 | 27680 | 27.585% |
| deepseek_r1_distill_qwen_1_5b_cats | math_qa | RelaCaTS-ES | 84.638% | 16.000 | 27680 | 27.585% |
| deepseek_r1_distill_qwen_1_5b_cats | math_qa | ASC | 85.936% | 5.070 | 6063 | 49.942% |
| deepseek_r1_distill_qwen_1_5b_cats | math_qa | RelaCaTS-ASC | 85.098% | 4.853 | 5743 | 50.466% |
| deepseek_r1_distill_qwen_1_5b_cats | math_qa | ESC | 86.061% | 16.000 | 27680 | 27.585% |
| deepseek_r1_distill_qwen_1_5b_cats | math_qa | RASC | 84.638% | 16.000 | 27680 | 27.585% |
| deepseek_r1_distill_qwen_1_5b_cats | arc_challenge | SC | 65.330% | 16.000 | 12309 | 16.651% |
| deepseek_r1_distill_qwen_1_5b_cats | arc_challenge | CISC | 65.439% | 16.000 | 12309 | 16.651% |
| deepseek_r1_distill_qwen_1_5b_cats | arc_challenge | Self-Certainty | 57.963% | 16.000 | 12309 | 16.651% |
| deepseek_r1_distill_qwen_1_5b_cats | arc_challenge | RelaCaTS-SC | 63.705% | 16.000 | 12309 | 16.651% |
| deepseek_r1_distill_qwen_1_5b_cats | arc_challenge | Best-of-N | 57.963% | 16.000 | 12309 | 16.651% |
| deepseek_r1_distill_qwen_1_5b_cats | arc_challenge | RelaCaTS-ES | 63.705% | 16.000 | 12309 | 16.651% |
| deepseek_r1_distill_qwen_1_5b_cats | arc_challenge | ASC | 63.922% | 8.277 | 6281 | 17.788% |
| deepseek_r1_distill_qwen_1_5b_cats | arc_challenge | RelaCaTS-ASC | 62.514% | 8.109 | 6151 | 17.822% |
| deepseek_r1_distill_qwen_1_5b_cats | arc_challenge | ESC | 65.330% | 16.000 | 12309 | 16.651% |
| deepseek_r1_distill_qwen_1_5b_cats | arc_challenge | RASC | 63.705% | 16.000 | 12309 | 16.651% |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | object_counting | SC | 72.193% | 16.000 | 2809 | 6.116% |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | object_counting | CISC | 72.193% | 16.000 | 2809 | 6.116% |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | object_counting | Self-Certainty | 58.824% | 16.000 | 2809 | 6.116% |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | object_counting | RelaCaTS-SC | 58.824% | 16.000 | 2809 | 6.116% |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | object_counting | Best-of-N | 58.824% | 16.000 | 2809 | 6.116% |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | object_counting | RelaCaTS-ES | 58.824% | 16.000 | 2809 | 6.116% |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | object_counting | ASC | 68.984% | 7.941 | 1395 | 6.061% |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | object_counting | RelaCaTS-ASC | 58.824% | 16.000 | 2809 | 6.116% |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | object_counting | ESC | 72.193% | 16.000 | 2809 | 6.116% |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | object_counting | RASC | 58.824% | 16.000 | 2809 | 6.116% |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | math_qa | SC | 85.098% | 16.000 | 27406 | 28.302% |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | math_qa | CISC | 85.098% | 16.000 | 27406 | 28.302% |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | math_qa | Self-Certainty | 83.591% | 16.000 | 27406 | 28.302% |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | math_qa | RelaCaTS-SC | 83.591% | 16.000 | 27406 | 28.302% |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | math_qa | Best-of-N | 83.591% | 16.000 | 27406 | 28.302% |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | math_qa | RelaCaTS-ES | 83.591% | 16.000 | 27406 | 28.302% |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | math_qa | ASC | 84.973% | 4.969 | 5253 | 55.749% |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | math_qa | RelaCaTS-ASC | 83.591% | 15.983 | 27379 | 28.297% |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | math_qa | ESC | 85.098% | 16.000 | 27406 | 28.302% |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | math_qa | RASC | 83.591% | 16.000 | 27406 | 28.302% |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | arc_challenge | SC | 66.414% | 16.000 | 12671 | 14.200% |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | arc_challenge | CISC | 66.414% | 16.000 | 12671 | 14.200% |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | arc_challenge | Self-Certainty | 59.805% | 16.000 | 12671 | 14.200% |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | arc_challenge | RelaCaTS-SC | 59.913% | 16.000 | 12671 | 14.200% |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | arc_challenge | Best-of-N | 59.805% | 16.000 | 12671 | 14.200% |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | arc_challenge | RelaCaTS-ES | 59.913% | 16.000 | 12671 | 14.200% |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | arc_challenge | ASC | 65.114% | 7.183 | 5493 | 17.149% |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | arc_challenge | RelaCaTS-ASC | 59.805% | 15.403 | 12191 | 14.251% |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | arc_challenge | ESC | 66.414% | 16.000 | 12671 | 14.200% |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | arc_challenge | RASC | 59.913% | 16.000 | 12671 | 14.200% |

## Comparisons

The old pools and `eval_outputs_v2` use different response generations and/or a full-test versus held-out denominator. Their deltas are diagnostic and are marked `directly_comparable=false`.

| Model | Dataset | Method | Reference | New accuracy | Reference accuracy | Delta |
|---|---|---|---|---:|---:|---:|
| qwen2_5_7b_instruct_cats | object_counting | SC | legacy_table2_results | 81.283% | 80.800% | +0.483 pp |
| qwen2_5_7b_instruct_cats | object_counting | CISC | legacy_table2_results | 82.353% | 80.400% | +1.953 pp |
| qwen2_5_7b_instruct_cats | object_counting | Self-Certainty | legacy_table2_results | 75.936% | 75.200% | +0.736 pp |
| qwen2_5_7b_instruct_cats | object_counting | RelaCaTS-SC | legacy_table2_results | 82.353% | 80.800% | +1.553 pp |
| qwen2_5_7b_instruct_cats | object_counting | Best-of-N | legacy_table2_results | 75.936% | 75.200% | +0.736 pp |
| qwen2_5_7b_instruct_cats | object_counting | RelaCaTS-ES | legacy_table2_results | 82.353% | 77.200% | +5.153 pp |
| qwen2_5_7b_instruct_cats | object_counting | ASC | legacy_table2_results | 81.283% | 80.400% | +0.883 pp |
| qwen2_5_7b_instruct_cats | object_counting | RelaCaTS-ASC | legacy_table2_results | 81.818% | 80.400% | +1.418 pp |
| qwen2_5_7b_instruct_cats | object_counting | ESC | legacy_table2_results | 81.283% | 80.800% | +0.483 pp |
| qwen2_5_7b_instruct_cats | object_counting | RASC | legacy_table2_results | 82.353% | 80.400% | +1.953 pp |
| qwen2_5_7b_instruct_cats | math_qa | SC | legacy_table2_results | 87.108% | 87.035% | +0.072 pp |
| qwen2_5_7b_instruct_cats | math_qa | CISC | legacy_table2_results | 87.652% | 87.605% | +0.047 pp |
| qwen2_5_7b_instruct_cats | math_qa | Self-Certainty | legacy_table2_results | 87.735% | 87.471% | +0.265 pp |
| qwen2_5_7b_instruct_cats | math_qa | RelaCaTS-SC | legacy_table2_results | 88.866% | 88.878% | -0.012 pp |
| qwen2_5_7b_instruct_cats | math_qa | Best-of-N | legacy_table2_results | 87.735% | 87.471% | +0.265 pp |
| qwen2_5_7b_instruct_cats | math_qa | RelaCaTS-ES | legacy_table2_results | 88.866% | 89.347% | -0.481 pp |
| qwen2_5_7b_instruct_cats | math_qa | ASC | legacy_table2_results | 87.149% | 86.399% | +0.751 pp |
| qwen2_5_7b_instruct_cats | math_qa | RelaCaTS-ASC | legacy_table2_results | 88.196% | 88.174% | +0.022 pp |
| qwen2_5_7b_instruct_cats | math_qa | ESC | legacy_table2_results | 87.108% | 87.169% | -0.062 pp |
| qwen2_5_7b_instruct_cats | math_qa | RASC | legacy_table2_results | 88.866% | 89.447% | -0.582 pp |
| qwen2_5_7b_instruct_cats | arc_challenge | SC | legacy_table2_results | 91.983% | 92.918% | -0.935 pp |
| qwen2_5_7b_instruct_cats | arc_challenge | CISC | legacy_table2_results | 91.983% | 93.003% | -1.021 pp |
| qwen2_5_7b_instruct_cats | arc_challenge | Self-Certainty | legacy_table2_results | 89.924% | 91.809% | -1.885 pp |
| qwen2_5_7b_instruct_cats | arc_challenge | RelaCaTS-SC | legacy_table2_results | 91.224% | 92.833% | -1.608 pp |
| qwen2_5_7b_instruct_cats | arc_challenge | Best-of-N | legacy_table2_results | 89.924% | 91.809% | -1.885 pp |
| qwen2_5_7b_instruct_cats | arc_challenge | RelaCaTS-ES | legacy_table2_results | 91.224% | 92.406% | -1.182 pp |
| qwen2_5_7b_instruct_cats | arc_challenge | ASC | legacy_table2_results | 91.441% | 93.174% | -1.733 pp |
| qwen2_5_7b_instruct_cats | arc_challenge | RelaCaTS-ASC | legacy_table2_results | 91.224% | 92.918% | -1.694 pp |
| qwen2_5_7b_instruct_cats | arc_challenge | ESC | legacy_table2_results | 91.983% | 93.003% | -1.021 pp |
| qwen2_5_7b_instruct_cats | arc_challenge | RASC | legacy_table2_results | 91.224% | 92.833% | -1.608 pp |
| qwen2_5_7b_instruct_relacats_v1 | object_counting | SC | eval_outputs_v2 | 79.679% | 80.214% | -0.535 pp |
| qwen2_5_7b_instruct_relacats_v1 | object_counting | SC | legacy_table2_results | 79.679% | 80.800% | -1.121 pp |
| qwen2_5_7b_instruct_relacats_v1 | object_counting | CISC | eval_outputs_v2 | 79.679% | 79.679% | +0.000 pp |
| qwen2_5_7b_instruct_relacats_v1 | object_counting | CISC | legacy_table2_results | 79.679% | 80.400% | -0.721 pp |
| qwen2_5_7b_instruct_relacats_v1 | object_counting | Self-Certainty | eval_outputs_v2 | 76.471% | 78.610% | -2.139 pp |
| qwen2_5_7b_instruct_relacats_v1 | object_counting | Self-Certainty | legacy_table2_results | 76.471% | 75.200% | +1.271 pp |
| qwen2_5_7b_instruct_relacats_v1 | object_counting | RelaCaTS-SC | eval_outputs_v2 | 79.679% | 79.679% | +0.000 pp |
| qwen2_5_7b_instruct_relacats_v1 | object_counting | RelaCaTS-SC | legacy_table2_results | 79.679% | 80.800% | -1.121 pp |
| qwen2_5_7b_instruct_relacats_v1 | object_counting | Best-of-N | eval_outputs_v2 | 76.471% | 78.610% | -2.139 pp |
| qwen2_5_7b_instruct_relacats_v1 | object_counting | Best-of-N | legacy_table2_results | 76.471% | 75.200% | +1.271 pp |
| qwen2_5_7b_instruct_relacats_v1 | object_counting | RelaCaTS-ES | eval_outputs_v2 | 79.679% | 79.679% | +0.000 pp |
| qwen2_5_7b_instruct_relacats_v1 | object_counting | RelaCaTS-ES | legacy_table2_results | 79.679% | 77.200% | +2.479 pp |
| qwen2_5_7b_instruct_relacats_v1 | object_counting | ASC | eval_outputs_v2 | 77.540% | 79.144% | -1.604 pp |
| qwen2_5_7b_instruct_relacats_v1 | object_counting | ASC | legacy_table2_results | 77.540% | 80.400% | -2.860 pp |
| qwen2_5_7b_instruct_relacats_v1 | object_counting | RelaCaTS-ASC | eval_outputs_v2 | 77.540% | 79.144% | -1.604 pp |
| qwen2_5_7b_instruct_relacats_v1 | object_counting | RelaCaTS-ASC | legacy_table2_results | 77.540% | 80.400% | -2.860 pp |
| qwen2_5_7b_instruct_relacats_v1 | object_counting | ESC | eval_outputs_v2 | 79.679% | 80.214% | -0.535 pp |
| qwen2_5_7b_instruct_relacats_v1 | object_counting | ESC | legacy_table2_results | 79.679% | 80.800% | -1.121 pp |
| qwen2_5_7b_instruct_relacats_v1 | object_counting | RASC | eval_outputs_v2 | 79.679% | 79.679% | +0.000 pp |
| qwen2_5_7b_instruct_relacats_v1 | object_counting | RASC | legacy_table2_results | 79.679% | 80.400% | -0.721 pp |
| qwen2_5_7b_instruct_relacats_v1 | math_qa | SC | eval_outputs_v2 | 87.987% | 87.987% | +0.000 pp |
| qwen2_5_7b_instruct_relacats_v1 | math_qa | SC | legacy_table2_results | 87.987% | 87.035% | +0.951 pp |
| qwen2_5_7b_instruct_relacats_v1 | math_qa | CISC | eval_outputs_v2 | 88.321% | 88.196% | +0.126 pp |
| qwen2_5_7b_instruct_relacats_v1 | math_qa | CISC | legacy_table2_results | 88.321% | 87.605% | +0.717 pp |
| qwen2_5_7b_instruct_relacats_v1 | math_qa | Self-Certainty | eval_outputs_v2 | 86.982% | 87.526% | -0.544 pp |
| qwen2_5_7b_instruct_relacats_v1 | math_qa | Self-Certainty | legacy_table2_results | 86.982% | 87.471% | -0.489 pp |
| qwen2_5_7b_instruct_relacats_v1 | math_qa | RelaCaTS-SC | eval_outputs_v2 | 88.740% | 88.405% | +0.335 pp |
| qwen2_5_7b_instruct_relacats_v1 | math_qa | RelaCaTS-SC | legacy_table2_results | 88.740% | 88.878% | -0.138 pp |
| qwen2_5_7b_instruct_relacats_v1 | math_qa | Best-of-N | eval_outputs_v2 | 86.982% | 87.526% | -0.544 pp |
| qwen2_5_7b_instruct_relacats_v1 | math_qa | Best-of-N | legacy_table2_results | 86.982% | 87.471% | -0.489 pp |
| qwen2_5_7b_instruct_relacats_v1 | math_qa | RelaCaTS-ES | eval_outputs_v2 | 88.740% | 88.405% | +0.335 pp |
| qwen2_5_7b_instruct_relacats_v1 | math_qa | RelaCaTS-ES | legacy_table2_results | 88.740% | 89.347% | -0.607 pp |
| qwen2_5_7b_instruct_relacats_v1 | math_qa | ASC | eval_outputs_v2 | 87.526% | 87.861% | -0.335 pp |
| qwen2_5_7b_instruct_relacats_v1 | math_qa | ASC | legacy_table2_results | 87.526% | 86.399% | +1.128 pp |
| qwen2_5_7b_instruct_relacats_v1 | math_qa | RelaCaTS-ASC | eval_outputs_v2 | 87.903% | 88.112% | -0.209 pp |
| qwen2_5_7b_instruct_relacats_v1 | math_qa | RelaCaTS-ASC | legacy_table2_results | 87.903% | 88.174% | -0.271 pp |
| qwen2_5_7b_instruct_relacats_v1 | math_qa | ESC | eval_outputs_v2 | 87.987% | 87.987% | +0.000 pp |
| qwen2_5_7b_instruct_relacats_v1 | math_qa | ESC | legacy_table2_results | 87.987% | 87.169% | +0.817 pp |
| qwen2_5_7b_instruct_relacats_v1 | math_qa | RASC | eval_outputs_v2 | 88.740% | 88.405% | +0.335 pp |
| qwen2_5_7b_instruct_relacats_v1 | math_qa | RASC | legacy_table2_results | 88.740% | 89.447% | -0.707 pp |
| qwen2_5_7b_instruct_relacats_v1 | arc_challenge | SC | eval_outputs_v2 | 91.874% | 93.824% | -1.950 pp |
| qwen2_5_7b_instruct_relacats_v1 | arc_challenge | SC | legacy_table2_results | 91.874% | 92.918% | -1.044 pp |
| qwen2_5_7b_instruct_relacats_v1 | arc_challenge | CISC | eval_outputs_v2 | 91.766% | 93.608% | -1.842 pp |
| qwen2_5_7b_instruct_relacats_v1 | arc_challenge | CISC | legacy_table2_results | 91.766% | 93.003% | -1.237 pp |
| qwen2_5_7b_instruct_relacats_v1 | arc_challenge | Self-Certainty | eval_outputs_v2 | 90.249% | 92.416% | -2.167 pp |
| qwen2_5_7b_instruct_relacats_v1 | arc_challenge | Self-Certainty | legacy_table2_results | 90.249% | 91.809% | -1.560 pp |
| qwen2_5_7b_instruct_relacats_v1 | arc_challenge | RelaCaTS-SC | eval_outputs_v2 | 91.658% | 93.499% | -1.842 pp |
| qwen2_5_7b_instruct_relacats_v1 | arc_challenge | RelaCaTS-SC | legacy_table2_results | 91.658% | 92.833% | -1.175 pp |
| qwen2_5_7b_instruct_relacats_v1 | arc_challenge | Best-of-N | eval_outputs_v2 | 90.249% | 92.416% | -2.167 pp |
| qwen2_5_7b_instruct_relacats_v1 | arc_challenge | Best-of-N | legacy_table2_results | 90.249% | 91.809% | -1.560 pp |
| qwen2_5_7b_instruct_relacats_v1 | arc_challenge | RelaCaTS-ES | eval_outputs_v2 | 91.658% | 93.499% | -1.842 pp |
| qwen2_5_7b_instruct_relacats_v1 | arc_challenge | RelaCaTS-ES | legacy_table2_results | 91.658% | 92.406% | -0.749 pp |
| qwen2_5_7b_instruct_relacats_v1 | arc_challenge | ASC | eval_outputs_v2 | 92.416% | 93.824% | -1.408 pp |
| qwen2_5_7b_instruct_relacats_v1 | arc_challenge | ASC | legacy_table2_results | 92.416% | 93.174% | -0.758 pp |
| qwen2_5_7b_instruct_relacats_v1 | arc_challenge | RelaCaTS-ASC | eval_outputs_v2 | 92.199% | 93.608% | -1.408 pp |
| qwen2_5_7b_instruct_relacats_v1 | arc_challenge | RelaCaTS-ASC | legacy_table2_results | 92.199% | 92.918% | -0.719 pp |
| qwen2_5_7b_instruct_relacats_v1 | arc_challenge | ESC | eval_outputs_v2 | 91.874% | 93.824% | -1.950 pp |
| qwen2_5_7b_instruct_relacats_v1 | arc_challenge | ESC | legacy_table2_results | 91.874% | 93.003% | -1.129 pp |
| qwen2_5_7b_instruct_relacats_v1 | arc_challenge | RASC | eval_outputs_v2 | 91.658% | 93.499% | -1.842 pp |
| qwen2_5_7b_instruct_relacats_v1 | arc_challenge | RASC | legacy_table2_results | 91.658% | 92.833% | -1.175 pp |
| llama3_1_8b_instruct_cats | object_counting | SC | legacy_table2_results | 77.540% | 73.600% | +3.940 pp |
| llama3_1_8b_instruct_cats | object_counting | CISC | legacy_table2_results | 76.471% | 76.000% | +0.471 pp |
| llama3_1_8b_instruct_cats | object_counting | Self-Certainty | legacy_table2_results | 78.075% | 70.400% | +7.675 pp |
| llama3_1_8b_instruct_cats | object_counting | RelaCaTS-SC | legacy_table2_results | 78.075% | 76.400% | +1.675 pp |
| llama3_1_8b_instruct_cats | object_counting | Best-of-N | legacy_table2_results | 78.075% | 70.400% | +7.675 pp |
| llama3_1_8b_instruct_cats | object_counting | RelaCaTS-ES | legacy_table2_results | 78.075% | 74.000% | +4.075 pp |
| llama3_1_8b_instruct_cats | object_counting | ASC | legacy_table2_results | 78.075% | 74.400% | +3.675 pp |
| llama3_1_8b_instruct_cats | object_counting | RelaCaTS-ASC | legacy_table2_results | 77.540% | 75.600% | +1.940 pp |
| llama3_1_8b_instruct_cats | object_counting | ESC | legacy_table2_results | 77.540% | 75.200% | +2.340 pp |
| llama3_1_8b_instruct_cats | object_counting | RASC | legacy_table2_results | 78.075% | 78.000% | +0.075 pp |
| llama3_1_8b_instruct_cats | math_qa | SC | legacy_table2_results | 83.005% | 84.288% | -1.283 pp |
| llama3_1_8b_instruct_cats | math_qa | CISC | legacy_table2_results | 84.261% | 85.561% | -1.300 pp |
| llama3_1_8b_instruct_cats | math_qa | Self-Certainty | legacy_table2_results | 81.457% | 82.513% | -1.056 pp |
| llama3_1_8b_instruct_cats | math_qa | RelaCaTS-SC | legacy_table2_results | 84.847% | 85.729% | -0.881 pp |
| llama3_1_8b_instruct_cats | math_qa | Best-of-N | legacy_table2_results | 81.457% | 82.513% | -1.056 pp |
| llama3_1_8b_instruct_cats | math_qa | RelaCaTS-ES | legacy_table2_results | 84.847% | 85.092% | -0.245 pp |
| llama3_1_8b_instruct_cats | math_qa | ASC | legacy_table2_results | 82.419% | 82.948% | -0.529 pp |
| llama3_1_8b_instruct_cats | math_qa | RelaCaTS-ASC | legacy_table2_results | 83.508% | 84.255% | -0.747 pp |
| llama3_1_8b_instruct_cats | math_qa | ESC | legacy_table2_results | 83.005% | 84.322% | -1.316 pp |
| llama3_1_8b_instruct_cats | math_qa | RASC | legacy_table2_results | 84.847% | 85.863% | -1.015 pp |
| llama3_1_8b_instruct_cats | arc_challenge | SC | legacy_table2_results | 89.274% | 89.078% | +0.196 pp |
| llama3_1_8b_instruct_cats | arc_challenge | CISC | legacy_table2_results | 89.707% | 89.932% | -0.224 pp |
| llama3_1_8b_instruct_cats | arc_challenge | Self-Certainty | legacy_table2_results | 87.432% | 88.396% | -0.964 pp |
| llama3_1_8b_instruct_cats | arc_challenge | RelaCaTS-SC | legacy_table2_results | 89.166% | 90.102% | -0.937 pp |
| llama3_1_8b_instruct_cats | arc_challenge | Best-of-N | legacy_table2_results | 87.432% | 88.396% | -0.964 pp |
| llama3_1_8b_instruct_cats | arc_challenge | RelaCaTS-ES | legacy_table2_results | 89.166% | 89.761% | -0.595 pp |
| llama3_1_8b_instruct_cats | arc_challenge | ASC | legacy_table2_results | 88.624% | 89.932% | -1.308 pp |
| llama3_1_8b_instruct_cats | arc_challenge | RelaCaTS-ASC | legacy_table2_results | 88.624% | 90.188% | -1.564 pp |
| llama3_1_8b_instruct_cats | arc_challenge | ESC | legacy_table2_results | 89.274% | 89.761% | -0.487 pp |
| llama3_1_8b_instruct_cats | arc_challenge | RASC | legacy_table2_results | 89.166% | 90.102% | -0.937 pp |
| llama3_1_8b_instruct_relacats_v1 | object_counting | SC | eval_outputs_v2 | 56.684% | 56.150% | +0.535 pp |
| llama3_1_8b_instruct_relacats_v1 | object_counting | SC | legacy_table2_results | 56.684% | 73.600% | -16.916 pp |
| llama3_1_8b_instruct_relacats_v1 | object_counting | CISC | eval_outputs_v2 | 57.754% | 56.684% | +1.070 pp |
| llama3_1_8b_instruct_relacats_v1 | object_counting | CISC | legacy_table2_results | 57.754% | 76.000% | -18.246 pp |
| llama3_1_8b_instruct_relacats_v1 | object_counting | Self-Certainty | eval_outputs_v2 | 57.754% | 56.150% | +1.604 pp |
| llama3_1_8b_instruct_relacats_v1 | object_counting | Self-Certainty | legacy_table2_results | 57.754% | 70.400% | -12.646 pp |
| llama3_1_8b_instruct_relacats_v1 | object_counting | RelaCaTS-SC | eval_outputs_v2 | 58.289% | 57.219% | +1.070 pp |
| llama3_1_8b_instruct_relacats_v1 | object_counting | RelaCaTS-SC | legacy_table2_results | 58.289% | 76.400% | -18.111 pp |
| llama3_1_8b_instruct_relacats_v1 | object_counting | Best-of-N | eval_outputs_v2 | 57.754% | 56.150% | +1.604 pp |
| llama3_1_8b_instruct_relacats_v1 | object_counting | Best-of-N | legacy_table2_results | 57.754% | 70.400% | -12.646 pp |
| llama3_1_8b_instruct_relacats_v1 | object_counting | RelaCaTS-ES | eval_outputs_v2 | 58.289% | 57.219% | +1.070 pp |
| llama3_1_8b_instruct_relacats_v1 | object_counting | RelaCaTS-ES | legacy_table2_results | 58.289% | 74.000% | -15.711 pp |
| llama3_1_8b_instruct_relacats_v1 | object_counting | ASC | eval_outputs_v2 | 56.150% | 56.150% | +0.000 pp |
| llama3_1_8b_instruct_relacats_v1 | object_counting | ASC | legacy_table2_results | 56.150% | 74.400% | -18.250 pp |
| llama3_1_8b_instruct_relacats_v1 | object_counting | RelaCaTS-ASC | eval_outputs_v2 | 57.219% | 56.684% | +0.535 pp |
| llama3_1_8b_instruct_relacats_v1 | object_counting | RelaCaTS-ASC | legacy_table2_results | 57.219% | 75.600% | -18.381 pp |
| llama3_1_8b_instruct_relacats_v1 | object_counting | ESC | eval_outputs_v2 | 56.684% | 56.150% | +0.535 pp |
| llama3_1_8b_instruct_relacats_v1 | object_counting | ESC | legacy_table2_results | 56.684% | 75.200% | -18.516 pp |
| llama3_1_8b_instruct_relacats_v1 | object_counting | RASC | eval_outputs_v2 | 58.289% | 57.219% | +1.070 pp |
| llama3_1_8b_instruct_relacats_v1 | object_counting | RASC | legacy_table2_results | 58.289% | 78.000% | -19.711 pp |
| llama3_1_8b_instruct_relacats_v1 | math_qa | SC | eval_outputs_v2 | 81.289% | 81.373% | -0.084 pp |
| llama3_1_8b_instruct_relacats_v1 | math_qa | SC | legacy_table2_results | 81.289% | 84.288% | -2.999 pp |
| llama3_1_8b_instruct_relacats_v1 | math_qa | CISC | eval_outputs_v2 | 82.252% | 82.545% | -0.293 pp |
| llama3_1_8b_instruct_relacats_v1 | math_qa | CISC | legacy_table2_results | 82.252% | 85.561% | -3.309 pp |
| llama3_1_8b_instruct_relacats_v1 | math_qa | Self-Certainty | eval_outputs_v2 | 78.694% | 77.355% | +1.339 pp |
| llama3_1_8b_instruct_relacats_v1 | math_qa | Self-Certainty | legacy_table2_results | 78.694% | 82.513% | -3.819 pp |
| llama3_1_8b_instruct_relacats_v1 | math_qa | RelaCaTS-SC | eval_outputs_v2 | 82.964% | 82.461% | +0.502 pp |
| llama3_1_8b_instruct_relacats_v1 | math_qa | RelaCaTS-SC | legacy_table2_results | 82.964% | 85.729% | -2.765 pp |
| llama3_1_8b_instruct_relacats_v1 | math_qa | Best-of-N | eval_outputs_v2 | 78.694% | 77.355% | +1.339 pp |
| llama3_1_8b_instruct_relacats_v1 | math_qa | Best-of-N | legacy_table2_results | 78.694% | 82.513% | -3.819 pp |
| llama3_1_8b_instruct_relacats_v1 | math_qa | RelaCaTS-ES | eval_outputs_v2 | 82.964% | 82.461% | +0.502 pp |
| llama3_1_8b_instruct_relacats_v1 | math_qa | RelaCaTS-ES | legacy_table2_results | 82.964% | 85.092% | -2.129 pp |
| llama3_1_8b_instruct_relacats_v1 | math_qa | ASC | eval_outputs_v2 | 80.075% | 80.452% | -0.377 pp |
| llama3_1_8b_instruct_relacats_v1 | math_qa | ASC | legacy_table2_results | 80.075% | 82.948% | -2.873 pp |
| llama3_1_8b_instruct_relacats_v1 | math_qa | RelaCaTS-ASC | eval_outputs_v2 | 81.457% | 81.247% | +0.209 pp |
| llama3_1_8b_instruct_relacats_v1 | math_qa | RelaCaTS-ASC | legacy_table2_results | 81.457% | 84.255% | -2.798 pp |
| llama3_1_8b_instruct_relacats_v1 | math_qa | ESC | eval_outputs_v2 | 81.289% | 81.373% | -0.084 pp |
| llama3_1_8b_instruct_relacats_v1 | math_qa | ESC | legacy_table2_results | 81.289% | 84.322% | -3.032 pp |
| llama3_1_8b_instruct_relacats_v1 | math_qa | RASC | eval_outputs_v2 | 82.964% | 82.461% | +0.502 pp |
| llama3_1_8b_instruct_relacats_v1 | math_qa | RASC | legacy_table2_results | 82.964% | 85.863% | -2.899 pp |
| llama3_1_8b_instruct_relacats_v1 | arc_challenge | SC | eval_outputs_v2 | 87.107% | 88.841% | -1.733 pp |
| llama3_1_8b_instruct_relacats_v1 | arc_challenge | SC | legacy_table2_results | 87.107% | 89.078% | -1.971 pp |
| llama3_1_8b_instruct_relacats_v1 | arc_challenge | CISC | eval_outputs_v2 | 86.566% | 89.057% | -2.492 pp |
| llama3_1_8b_instruct_relacats_v1 | arc_challenge | CISC | legacy_table2_results | 86.566% | 89.932% | -3.366 pp |
| llama3_1_8b_instruct_relacats_v1 | arc_challenge | Self-Certainty | eval_outputs_v2 | 83.857% | 84.724% | -0.867 pp |
| llama3_1_8b_instruct_relacats_v1 | arc_challenge | Self-Certainty | legacy_table2_results | 83.857% | 88.396% | -4.539 pp |
| llama3_1_8b_instruct_relacats_v1 | arc_challenge | RelaCaTS-SC | eval_outputs_v2 | 86.566% | 88.624% | -2.059 pp |
| llama3_1_8b_instruct_relacats_v1 | arc_challenge | RelaCaTS-SC | legacy_table2_results | 86.566% | 90.102% | -3.537 pp |
| llama3_1_8b_instruct_relacats_v1 | arc_challenge | Best-of-N | eval_outputs_v2 | 83.857% | 84.724% | -0.867 pp |
| llama3_1_8b_instruct_relacats_v1 | arc_challenge | Best-of-N | legacy_table2_results | 83.857% | 88.396% | -4.539 pp |
| llama3_1_8b_instruct_relacats_v1 | arc_challenge | RelaCaTS-ES | eval_outputs_v2 | 86.566% | 88.624% | -2.059 pp |
| llama3_1_8b_instruct_relacats_v1 | arc_challenge | RelaCaTS-ES | legacy_table2_results | 86.566% | 89.761% | -3.196 pp |
| llama3_1_8b_instruct_relacats_v1 | arc_challenge | ASC | eval_outputs_v2 | 86.566% | 88.407% | -1.842 pp |
| llama3_1_8b_instruct_relacats_v1 | arc_challenge | ASC | legacy_table2_results | 86.566% | 89.932% | -3.366 pp |
| llama3_1_8b_instruct_relacats_v1 | arc_challenge | RelaCaTS-ASC | eval_outputs_v2 | 86.782% | 88.191% | -1.408 pp |
| llama3_1_8b_instruct_relacats_v1 | arc_challenge | RelaCaTS-ASC | legacy_table2_results | 86.782% | 90.188% | -3.405 pp |
| llama3_1_8b_instruct_relacats_v1 | arc_challenge | ESC | eval_outputs_v2 | 87.107% | 88.841% | -1.733 pp |
| llama3_1_8b_instruct_relacats_v1 | arc_challenge | ESC | legacy_table2_results | 87.107% | 89.761% | -2.654 pp |
| llama3_1_8b_instruct_relacats_v1 | arc_challenge | RASC | eval_outputs_v2 | 86.566% | 88.624% | -2.059 pp |
| llama3_1_8b_instruct_relacats_v1 | arc_challenge | RASC | legacy_table2_results | 86.566% | 90.102% | -3.537 pp |
| deepseek_r1_distill_qwen_1_5b_cats | object_counting | SC | legacy_table2_results | 73.797% | 73.200% | +0.597 pp |
| deepseek_r1_distill_qwen_1_5b_cats | object_counting | CISC | legacy_table2_results | 72.727% | 72.400% | +0.327 pp |
| deepseek_r1_distill_qwen_1_5b_cats | object_counting | Self-Certainty | legacy_table2_results | 55.615% | 56.000% | -0.385 pp |
| deepseek_r1_distill_qwen_1_5b_cats | object_counting | RelaCaTS-SC | legacy_table2_results | 67.914% | 72.400% | -4.486 pp |
| deepseek_r1_distill_qwen_1_5b_cats | object_counting | Best-of-N | legacy_table2_results | 55.615% | 56.000% | -0.385 pp |
| deepseek_r1_distill_qwen_1_5b_cats | object_counting | RelaCaTS-ES | legacy_table2_results | 67.914% | 57.600% | +10.314 pp |
| deepseek_r1_distill_qwen_1_5b_cats | object_counting | ASC | legacy_table2_results | 72.727% | 70.000% | +2.727 pp |
| deepseek_r1_distill_qwen_1_5b_cats | object_counting | RelaCaTS-ASC | legacy_table2_results | 64.706% | 70.800% | -6.094 pp |
| deepseek_r1_distill_qwen_1_5b_cats | object_counting | ESC | legacy_table2_results | 73.797% | 70.000% | +3.797 pp |
| deepseek_r1_distill_qwen_1_5b_cats | object_counting | RASC | legacy_table2_results | 67.914% | 66.800% | +1.114 pp |
| deepseek_r1_distill_qwen_1_5b_cats | math_qa | SC | legacy_table2_results | 86.061% | 91.832% | -5.771 pp |
| deepseek_r1_distill_qwen_1_5b_cats | math_qa | CISC | legacy_table2_results | 85.977% | 91.795% | -5.818 pp |
| deepseek_r1_distill_qwen_1_5b_cats | math_qa | Self-Certainty | legacy_table2_results | 81.750% | 91.317% | -9.567 pp |
| deepseek_r1_distill_qwen_1_5b_cats | math_qa | RelaCaTS-SC | legacy_table2_results | 84.638% | 91.795% | -7.158 pp |
| deepseek_r1_distill_qwen_1_5b_cats | math_qa | Best-of-N | legacy_table2_results | 81.750% | 91.317% | -9.567 pp |
| deepseek_r1_distill_qwen_1_5b_cats | math_qa | RelaCaTS-ES | legacy_table2_results | 84.638% | 91.722% | -7.084 pp |
| deepseek_r1_distill_qwen_1_5b_cats | math_qa | ASC | legacy_table2_results | 85.936% | 91.575% | -5.639 pp |
| deepseek_r1_distill_qwen_1_5b_cats | math_qa | RelaCaTS-ASC | legacy_table2_results | 85.098% | 91.538% | -6.440 pp |
| deepseek_r1_distill_qwen_1_5b_cats | math_qa | ESC | legacy_table2_results | 86.061% | 91.832% | -5.771 pp |
| deepseek_r1_distill_qwen_1_5b_cats | math_qa | RASC | legacy_table2_results | 84.638% | 91.795% | -7.158 pp |
| deepseek_r1_distill_qwen_1_5b_cats | arc_challenge | SC | legacy_table2_results | 65.330% | 67.065% | -1.734 pp |
| deepseek_r1_distill_qwen_1_5b_cats | arc_challenge | CISC | legacy_table2_results | 65.439% | 67.662% | -2.223 pp |
| deepseek_r1_distill_qwen_1_5b_cats | arc_challenge | Self-Certainty | legacy_table2_results | 57.963% | 63.140% | -5.177 pp |
| deepseek_r1_distill_qwen_1_5b_cats | arc_challenge | RelaCaTS-SC | legacy_table2_results | 63.705% | 67.235% | -3.530 pp |
| deepseek_r1_distill_qwen_1_5b_cats | arc_challenge | Best-of-N | legacy_table2_results | 57.963% | 63.140% | -5.177 pp |
| deepseek_r1_distill_qwen_1_5b_cats | arc_challenge | RelaCaTS-ES | legacy_table2_results | 63.705% | 65.529% | -1.824 pp |
| deepseek_r1_distill_qwen_1_5b_cats | arc_challenge | ASC | legacy_table2_results | 63.922% | 66.212% | -2.290 pp |
| deepseek_r1_distill_qwen_1_5b_cats | arc_challenge | RelaCaTS-ASC | legacy_table2_results | 62.514% | 66.126% | -3.613 pp |
| deepseek_r1_distill_qwen_1_5b_cats | arc_challenge | ESC | legacy_table2_results | 65.330% | 67.833% | -2.502 pp |
| deepseek_r1_distill_qwen_1_5b_cats | arc_challenge | RASC | legacy_table2_results | 63.705% | 66.553% | -2.848 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | object_counting | SC | eval_outputs_v2 | 72.193% | 71.123% | +1.070 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | object_counting | SC | legacy_table2_results | 72.193% | 73.200% | -1.007 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | object_counting | CISC | eval_outputs_v2 | 72.193% | 69.519% | +2.674 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | object_counting | CISC | legacy_table2_results | 72.193% | 72.400% | -0.207 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | object_counting | Self-Certainty | eval_outputs_v2 | 58.824% | 51.872% | +6.952 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | object_counting | Self-Certainty | legacy_table2_results | 58.824% | 56.000% | +2.824 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | object_counting | RelaCaTS-SC | eval_outputs_v2 | 58.824% | 68.984% | -10.160 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | object_counting | RelaCaTS-SC | legacy_table2_results | 58.824% | 72.400% | -13.576 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | object_counting | Best-of-N | eval_outputs_v2 | 58.824% | 51.872% | +6.952 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | object_counting | Best-of-N | legacy_table2_results | 58.824% | 56.000% | +2.824 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | object_counting | RelaCaTS-ES | eval_outputs_v2 | 58.824% | 68.984% | -10.160 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | object_counting | RelaCaTS-ES | legacy_table2_results | 58.824% | 57.600% | +1.224 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | object_counting | ASC | eval_outputs_v2 | 68.984% | 70.588% | -1.604 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | object_counting | ASC | legacy_table2_results | 68.984% | 70.000% | -1.016 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | object_counting | RelaCaTS-ASC | eval_outputs_v2 | 58.824% | 68.449% | -9.626 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | object_counting | RelaCaTS-ASC | legacy_table2_results | 58.824% | 70.800% | -11.976 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | object_counting | ESC | eval_outputs_v2 | 72.193% | 71.123% | +1.070 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | object_counting | ESC | legacy_table2_results | 72.193% | 70.000% | +2.193 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | object_counting | RASC | eval_outputs_v2 | 58.824% | 68.984% | -10.160 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | object_counting | RASC | legacy_table2_results | 58.824% | 66.800% | -7.976 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | math_qa | SC | eval_outputs_v2 | 85.098% | 80.326% | +4.772 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | math_qa | SC | legacy_table2_results | 85.098% | 91.832% | -6.734 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | math_qa | CISC | eval_outputs_v2 | 85.098% | 80.578% | +4.521 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | math_qa | CISC | legacy_table2_results | 85.098% | 91.795% | -6.697 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | math_qa | Self-Certainty | eval_outputs_v2 | 83.591% | 79.824% | +3.767 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | math_qa | Self-Certainty | legacy_table2_results | 83.591% | 91.317% | -7.726 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | math_qa | RelaCaTS-SC | eval_outputs_v2 | 83.591% | 80.745% | +2.846 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | math_qa | RelaCaTS-SC | legacy_table2_results | 83.591% | 91.795% | -8.204 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | math_qa | Best-of-N | eval_outputs_v2 | 83.591% | 79.824% | +3.767 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | math_qa | Best-of-N | legacy_table2_results | 83.591% | 91.317% | -7.726 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | math_qa | RelaCaTS-ES | eval_outputs_v2 | 83.591% | 80.745% | +2.846 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | math_qa | RelaCaTS-ES | legacy_table2_results | 83.591% | 91.722% | -8.130 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | math_qa | ASC | eval_outputs_v2 | 84.973% | 80.243% | +4.730 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | math_qa | ASC | legacy_table2_results | 84.973% | 91.575% | -6.602 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | math_qa | RelaCaTS-ASC | eval_outputs_v2 | 83.591% | 80.578% | +3.014 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | math_qa | RelaCaTS-ASC | legacy_table2_results | 83.591% | 91.538% | -7.946 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | math_qa | ESC | eval_outputs_v2 | 85.098% | 80.326% | +4.772 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | math_qa | ESC | legacy_table2_results | 85.098% | 91.832% | -6.734 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | math_qa | RASC | eval_outputs_v2 | 83.591% | 80.745% | +2.846 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | math_qa | RASC | legacy_table2_results | 83.591% | 91.795% | -8.204 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | arc_challenge | SC | eval_outputs_v2 | 66.414% | 66.847% | -0.433 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | arc_challenge | SC | legacy_table2_results | 66.414% | 67.065% | -0.651 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | arc_challenge | CISC | eval_outputs_v2 | 66.414% | 66.197% | +0.217 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | arc_challenge | CISC | legacy_table2_results | 66.414% | 67.662% | -1.248 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | arc_challenge | Self-Certainty | eval_outputs_v2 | 59.805% | 60.130% | -0.325 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | arc_challenge | Self-Certainty | legacy_table2_results | 59.805% | 63.140% | -3.335 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | arc_challenge | RelaCaTS-SC | eval_outputs_v2 | 59.913% | 65.005% | -5.092 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | arc_challenge | RelaCaTS-SC | legacy_table2_results | 59.913% | 67.235% | -7.322 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | arc_challenge | Best-of-N | eval_outputs_v2 | 59.805% | 60.130% | -0.325 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | arc_challenge | Best-of-N | legacy_table2_results | 59.805% | 63.140% | -3.335 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | arc_challenge | RelaCaTS-ES | eval_outputs_v2 | 59.913% | 65.005% | -5.092 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | arc_challenge | RelaCaTS-ES | legacy_table2_results | 59.913% | 65.529% | -5.616 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | arc_challenge | ASC | eval_outputs_v2 | 65.114% | 66.089% | -0.975 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | arc_challenge | ASC | legacy_table2_results | 65.114% | 66.212% | -1.098 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | arc_challenge | RelaCaTS-ASC | eval_outputs_v2 | 59.805% | 65.547% | -5.742 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | arc_challenge | RelaCaTS-ASC | legacy_table2_results | 59.805% | 66.126% | -6.321 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | arc_challenge | ESC | eval_outputs_v2 | 66.414% | 66.847% | -0.433 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | arc_challenge | ESC | legacy_table2_results | 66.414% | 67.833% | -1.419 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | arc_challenge | RASC | eval_outputs_v2 | 59.913% | 65.005% | -5.092 pp |
| deepseek_r1_distill_qwen_1_5b_relacats_v1 | arc_challenge | RASC | legacy_table2_results | 59.913% | 66.553% | -6.640 pp |

## Artifact audit

- Validated model/dataset bundles: `18`
- Validated questions: `26442`
- Validated response/confidence samples: `846144`
- Every question has exactly `32` response and confidence records; sample IDs are paired one-to-one; non-finite numeric values are rejected.
