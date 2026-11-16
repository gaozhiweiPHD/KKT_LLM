# Measured trajectory case studies

Examples prefer one measured ID success and one failure from each reward variant A/B/C, selecting the first distinct problem IDs in file order. Other models/baselines are fallbacks only. These are illustrative cases, not a random sample for estimating rates.

## Success 1: qwen_grpo_A / test_id-00000
Seed 11; reason kkt_success; 10 actions; invalid=0; cycle=False.
Final normalized residuals: stationarity=5.1e-17, primal=4.9e-17, dual=0, complementarity=1.07e-17.

| Step | Action | Active set | KKT merit |
|---|---|---|---|
| 0 | initial | [] | 0.5048 |
| 1 | ADD(13) | [13] | 0.2538 |
| 2 | ADD(16) | [13, 16] | 0.09669 |
| 3 | ADD(25) | [13, 16, 25] | 0.09375 |
| 4 | ADD(30) | [13, 16, 25, 30] | 0.03096 |
| 5 | ADD(36) | [13, 16, 25, 30, 36] | 0.01641 |
| 6 | ADD(27) | [13, 16, 25, 27, 30, 36] | 0.01475 |
| 7 | ADD(29) | [13, 16, 25, 27, 29, 30, 36] | 0.0222 |
| 8 | ADD(20) | [13, 16, 20, 25, 27, 29, 30, 36] | 0.00597 |
| 9 | ADD(21) | [13, 16, 20, 21, 25, 27, 29, 30, 36] | 1.106e-16 |
| 10 | STOP | [13, 16, 20, 21, 25, 27, 29, 30, 36] | 1.106e-16 |

The final explicit STOP produced a valid primal-dual certificate; exact oracle-set equality was not required.
Merit increased on 1 transitions; DROP actions=0. Initial merit=0.5048, final merit=1.106e-16. An intermediate increase is not itself a failure: feasibility and multiplier repair can trade off across steps.
Generated tokens=56; model inference share=0.2557s; numerical solve time=0.0105s; episode latency=13.3562s.

## Success 2: qwen_grpo_B / test_id-00001
Seed 11; reason kkt_success; 20 actions; invalid=0; cycle=True.
Final normalized residuals: stationarity=3.33e-17, primal=3.64e-17, dual=0, complementarity=1.09e-18.

| Step | Action | Active set | KKT merit |
|---|---|---|---|
| 0 | initial | [] | 0.1903 |
| 1 | ADD(16) | [16] | 0.09491 |
| 2 | ADD(25) | [16, 25] | 0.05427 |
| 3 | DROP(25) | [16] | 0.09491 |
| 4 | ADD(25) | [16, 25] | 0.05427 |
| 5 | ADD(20) | [16, 20, 25] | 0.06014 |
| 6 | ADD(5) | [5, 16, 20, 25] | 0.03062 |
| 7 | ADD(18) | [5, 16, 18, 20, 25] | 0.01785 |
| 8 | ADD(26) | [5, 16, 18, 20, 25, 26] | 0.01126 |
| 9 | ADD(13) | [5, 13, 16, 18, 20, 25, 26] | 0.006474 |
| 10 | ADD(24) | [5, 13, 16, 18, 20, 24, 25, 26] | 0.006963 |
| 11 | ADD(11) | [5, 11, 13, 16, 18, 20, 24, 25, 26] | 0.007617 |
| 12 | DROP(18) | [5, 11, 13, 16, 20, 24, 25, 26] | 0.005175 |
| 13 | ADD(30) | [5, 11, 13, 16, 20, 24, 25, 26, 30] | 0.006788 |
| 14 | DROP(20) | [5, 11, 13, 16, 24, 25, 26, 30] | 0.002393 |
| 15 | ADD(22) | [5, 11, 13, 16, 22, 24, 25, 26, 30] | 0.01529 |
| 16 | DROP(26) | [5, 11, 13, 16, 22, 24, 25, 30] | 0.002515 |
| 17 | DROP(13) | [5, 11, 16, 22, 24, 25, 30] | 0.001465 |
| 18 | DROP(16) | [5, 11, 22, 24, 25, 30] | 0.001048 |
| 19 | ADD(27) | [5, 11, 22, 24, 25, 27, 30] | 7.081e-17 |
| 20 | STOP | [5, 11, 22, 24, 25, 27, 30] | 7.081e-17 |

Repeated active sets reveal cycling in this repair heuristic; no global convergence is claimed.
Merit increased on 6 transitions; DROP actions=6. Initial merit=0.1903, final merit=7.081e-17. An intermediate increase is not itself a failure: feasibility and multiplier repair can trade off across steps.
Generated tokens=115; model inference share=0.7274s; numerical solve time=0.0152s; episode latency=18.8804s.

## Success 3: qwen_grpo_C / test_id-00002
Seed 11; reason kkt_success; 9 actions; invalid=0; cycle=False.
Final normalized residuals: stationarity=6.23e-17, primal=1.4e-17, dual=1.65e-06, complementarity=4.12e-19.

| Step | Action | Active set | KKT merit |
|---|---|---|---|
| 0 | initial | [] | 0.02985 |
| 1 | ADD(23) | [23] | 0.03022 |
| 2 | ADD(4) | [4, 23] | 0.02886 |
| 3 | ADD(9) | [4, 9, 23] | 0.01197 |
| 4 | ADD(16) | [4, 9, 16, 23] | 0.01664 |
| 5 | ADD(24) | [4, 9, 16, 23, 24] | 0.005399 |
| 6 | ADD(17) | [4, 9, 16, 17, 23, 24] | 0.004994 |
| 7 | ADD(20) | [4, 9, 16, 17, 20, 23, 24] | 0.004085 |
| 8 | ADD(22) | [4, 9, 16, 17, 20, 22, 23, 24] | 1.654e-06 |
| 9 | STOP | [4, 9, 16, 17, 20, 22, 23, 24] | 1.654e-06 |

The final explicit STOP produced a valid primal-dual certificate; exact oracle-set equality was not required.
Merit increased on 2 transitions; DROP actions=0. Initial merit=0.02985, final merit=1.654e-06. An intermediate increase is not itself a failure: feasibility and multiplier repair can trade off across steps.
Generated tokens=48; model inference share=0.2248s; numerical solve time=0.0052s; episode latency=12.4381s.

## Failure 1: qwen_grpo_A / test_id-00013
Seed 11; reason max_steps; 20 actions; invalid=0; cycle=False.
Final normalized residuals: stationarity=3.24e-17, primal=0.00117, dual=0, complementarity=2.58e-18.

| Step | Action | Active set | KKT merit |
|---|---|---|---|
| 0 | initial | [] | 0.2109 |
| 1 | ADD(22) | [22] | 0.07603 |
| 2 | ADD(32) | [22, 32] | 0.06949 |
| 3 | ADD(21) | [21, 22, 32] | 0.02903 |
| 4 | ADD(28) | [21, 22, 28, 32] | 0.04004 |
| 5 | ADD(12) | [12, 21, 22, 28, 32] | 0.008461 |
| 6 | ADD(13) | [12, 13, 21, 22, 28, 32] | 0.01212 |
| 7 | ADD(30) | [12, 13, 21, 22, 28, 30, 32] | 0.0169 |
| 8 | DROP(22) | [12, 13, 21, 28, 30, 32] | 0.114 |
| 9 | DROP(12) | [13, 21, 28, 30, 32] | 0.1361 |
| 10 | DROP(30) | [13, 21, 28, 32] | 0.1535 |
| 11 | DROP(32) | [13, 21, 28] | 0.1508 |
| 12 | ADD(22) | [13, 21, 22, 28] | 0.03928 |
| 13 | ADD(30) | [13, 21, 22, 28, 30] | 0.007251 |
| 14 | ADD(5) | [5, 13, 21, 22, 28, 30] | 0.004447 |
| 15 | ADD(11) | [5, 11, 13, 21, 22, 28, 30] | 0.003741 |
| 16 | ADD(20) | [5, 11, 13, 20, 21, 22, 28, 30] | 0.004674 |
| 17 | ADD(14) | [5, 11, 13, 14, 20, 21, 22, 28, 30] | 0.002984 |
| 18 | DROP(30) | [5, 11, 13, 14, 20, 21, 22, 28] | 0.0017 |
| 19 | ADD(9) | [5, 9, 11, 13, 14, 20, 21, 22, 28] | 0.001615 |
| 20 | ADD(19) | [5, 9, 11, 13, 14, 19, 20, 21, 22, 28] | 0.001165 |

The policy exhausted its budget before issuing a certified STOP.
Merit increased on 7 transitions; DROP actions=5. Initial merit=0.2109, final merit=0.001165. An intermediate increase is not itself a failure: feasibility and multiplier repair can trade off across steps.
Generated tokens=118; model inference share=0.6774s; numerical solve time=0.0130s; episode latency=18.9086s.

## Failure 2: qwen_grpo_B / test_id-00017
Seed 11; reason incorrect_stop; 11 actions; invalid=0; cycle=True.
Final normalized residuals: stationarity=3.23e-17, primal=1.79e-17, dual=0.000178, complementarity=8.68e-20.

| Step | Action | Active set | KKT merit |
|---|---|---|---|
| 0 | initial | [] | 0.3354 |
| 1 | ADD(23) | [23] | 0.06962 |
| 2 | ADD(1) | [1, 23] | 0.01694 |
| 3 | ADD(21) | [1, 21, 23] | 0.009134 |
| 4 | ADD(2) | [1, 2, 21, 23] | 0.004889 |
| 5 | ADD(15) | [1, 2, 15, 21, 23] | 0.001634 |
| 6 | ADD(10) | [1, 2, 10, 15, 21, 23] | 0.001098 |
| 7 | DROP(2) | [1, 10, 15, 21, 23] | 0.001051 |
| 8 | ADD(16) | [1, 10, 15, 16, 21, 23] | 0.0001777 |
| 9 | DROP(10) | [1, 15, 16, 21, 23] | 0.002733 |
| 10 | ADD(10) | [1, 10, 15, 16, 21, 23] | 0.0001777 |
| 11 | STOP | [1, 10, 15, 16, 21, 23] | 0.0001777 |

The policy stopped before the independent verifier accepted all four residuals.
Merit increased on 1 transitions; DROP actions=2. Initial merit=0.3354, final merit=0.0001777. An intermediate increase is not itself a failure: feasibility and multiplier repair can trade off across steps.
Generated tokens=59; model inference share=0.2860s; numerical solve time=0.0065s; episode latency=13.9363s.

## Failure 3: qwen_grpo_C / test_id-00026
Seed 11; reason max_steps; 20 actions; invalid=8; cycle=False.
Final normalized residuals: stationarity=4.18e-17, primal=1.92e-06, dual=0, complementarity=1.64e-17.

| Step | Action | Active set | KKT merit |
|---|---|---|---|
| 0 | initial | [] | 0.2055 |
| 1 | ADD(8) | [8] | 0.1949 |
| 2 | ADD(28) | [8, 28] | 0.1634 |
| 3 | ADD(37) | [8, 28, 37] | 0.1203 |
| 4 | ADD(25) | [8, 25, 28, 37] | 0.06879 |
| 5 | ADD(18) | [8, 18, 25, 28, 37] | 0.04809 |
| 6 | ADD(23) | [8, 18, 23, 25, 28, 37] | 0.04673 |
| 7 | ADD(27) | [8, 18, 23, 25, 27, 28, 37] | 0.02036 |
| 8 | ADD(26) | [8, 18, 23, 25, 26, 27, 28, 37] | 0.007409 |
| 9 | ADD(33) | [8, 18, 23, 25, 26, 27, 28, 33, 37] | 0.006605 |
| 10 | ADD(10) | [8, 10, 18, 23, 25, 26, 27, 28, 33, 37] | 0.005326 |
| 11 | ADD(38) | [8, 10, 18, 23, 25, 26, 27, 28, 33, 37, 38] | 0.003181 |
| 12 | ADD(7) | [7, 8, 10, 18, 23, 25, 26, 27, 28, 33, 37, 38] | 1.917e-06 |
| 13 | ADD(1) | [7, 8, 10, 18, 23, 25, 26, 27, 28, 33, 37, 38] | 1.917e-06 |
| 14 | ADD(1) | [7, 8, 10, 18, 23, 25, 26, 27, 28, 33, 37, 38] | 1.917e-06 |
| 15 | ADD(1) | [7, 8, 10, 18, 23, 25, 26, 27, 28, 33, 37, 38] | 1.917e-06 |
| 16 | ADD(1) | [7, 8, 10, 18, 23, 25, 26, 27, 28, 33, 37, 38] | 1.917e-06 |
| 17 | ADD(1) | [7, 8, 10, 18, 23, 25, 26, 27, 28, 33, 37, 38] | 1.917e-06 |
| 18 | ADD(1) | [7, 8, 10, 18, 23, 25, 26, 27, 28, 33, 37, 38] | 1.917e-06 |
| 19 | ADD(1) | [7, 8, 10, 18, 23, 25, 26, 27, 28, 33, 37, 38] | 1.917e-06 |
| 20 | ADD(1) | [7, 8, 10, 18, 23, 25, 26, 27, 28, 33, 37, 38] | 1.917e-06 |

Rejected actions consumed the decision budget without changing the active set.
Merit increased on 0 transitions; DROP actions=0. Initial merit=0.2055, final merit=1.917e-06. An intermediate increase is not itself a failure: feasibility and multiplier repair can trade off across steps.
Generated tokens=110; model inference share=0.7003s; numerical solve time=0.0121s; episode latency=18.3632s.
