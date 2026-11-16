# Method and evaluation contract

## Problem and certificate

For a strictly convex QP, minimize $f(x)=\tfrac12x^\top Qx+c^\top x$ subject to $Ax\le b$, with symmetric positive definite $Q$. The KKT conditions are

$$Qx+c+A^\top\lambda=0,\quad Ax\le b,\quad\lambda\ge0,\quad\lambda\odot(Ax-b)=0.$$

If these hold exactly, then for any feasible $y$,

$$f(y)-f(x)=\lambda^\top(b-Ay)+\tfrac12(y-x)^\top Q(y-x)\ge0.$$

Positive definiteness makes the primal optimum unique. The numerical verifier instead accepts four normalized residuals individually at tolerance $10^{-5}$; that is an approximate certificate, not an exact proof or a universal absolute solution-error bound.

Set $X=\max(1,\|x\|_\infty)$, $a_i=\|A_i\|_1$, $G=\|Q\|_\infty X+\|c\|_\infty$, $P_i=a_iX+|b_i|$. The residuals are

$$r_s=\frac{\|Qx+c+A^\top\lambda\|_\infty}{G},\quad
r_p=\max_i\frac{(A_ix-b_i)_+}{P_i},$$
$$r_d=\max_i\frac{(-\lambda_i)_+a_i}{G},\quad
r_c=\max_i\frac{|\lambda_i(A_ix-b_i)|}{GX}.$$

The code floors denominators at the smallest positive normal float and rejects nonfinite candidates. Positive objective scaling and positive constraint-row scaling, with matching multiplier transformations, preserve these normalized residuals up to floating-point effects. Raw residuals are also recorded. See `kkt_agent/kkt_verifier.py`.

## Policy / solver boundary

For active indices $I$, solve the equality-constrained QP using Cholesky factors of $Q$ and a scaled Schur complement. Algebraically,

$$S\lambda_I=-b_I-A_IQ^{-1}c,\quad S=A_IQ^{-1}A_I^\top,\quad x=-Q^{-1}(c+A_I^\top\lambda_I).$$

The implementation uses linear solves, not explicit inverses. It checks rank, conditioning and equality residuals. An illegal or numerically inadmissible action consumes a step and leaves the state unchanged. A candidate subproblem solution need not be feasible for the original inequalities.

The compact policy sees the active set, normalized violated constraints, active multipliers, certificate residuals, step count and last error. It does not see oracle solutions or the full constraint geometry. The environment persists between actions; the compact policy is reactive to the latest state. Full-matrix conversation mode is available but was not evaluated in the recorded experiment.

Each episode permits 20 actions including STOP. Success requires explicit STOP and a passing certificate. Incorrect STOP, exhausted budget, cycles and numerical errors are distinct outcomes. A valid candidate without STOP is not agent success. Repeated active sets terminate after four visits. This heuristic repair environment has no general global-convergence guarantee; the separate classical baseline implements phase-I feasibility, feasible directions and blocking-constraint line search.

## Data and oracle isolation

The generator samples a known primal/dual optimum, positive inactive slacks, full-row-rank active constraints and a positive definite Hessian, then constructs $b$ and $c$ to satisfy KKT. Every saved problem is independently checked with OSQP. Public QPs, oracle solutions, difficulty metadata and numerical audits are stored separately.

| Split | Count | Seeds | Scope |
|---|---:|---|---|
| train | 5,000 | 10,000,000 + index | n=12, m=30–40, condition 10/100/1000 |
| validation | 500 | 20,000,000 + index | Same nominal distribution |
| test_id | 1,000 | 30,000,000 + index | Same nominal distribution |
| test_ood | 500 | 40,000,000 + index | n=12 with condition 10,000; or n=16 with m=30–40 / 45–50 |

Families are standard, small-slack, nearly-active and correlated. Offline teacher trajectories and curriculum construction may read training oracles. Online models receive public numerical observations and their initial repair state. Evaluation is always cold start; difficulty metadata is attached for subsequent stratification.

## SFT and GRPO

Qwen3-0.6B is pinned to `configs/model_revision.txt`. LoRA adapts q/k/v/o projections with rank 8, alpha 16 and dropout 0. SFT uses completion-only cross-entropy in TRL, learning action and EOS tokens from teacher trajectories. The default run uses the first 12,000 supervised states, 200 optimizer steps, batch size 1 and gradient accumulation 4.

The main GRPO implementation is custom (`kkt_agent/grpo.py`). For each sampled QP and initial active set it generates four complete trajectories before updating, with temperature 0.7, top_p=1, top_k=0 and at most 12 new tokens per action. Observation tokens are masked out of the objective. Let $R_g$ be terminal reward and

$$\hat A_g=\frac{R_g-\bar R}{\operatorname{std}_{pop}(R)+10^{-4}}.$$

The loss averages the clipped policy surrogate over tokens within each trajectory, then over trajectories, with clip range [0.8, 1.2] and KL coefficient zero. One on-policy update uses a detached replay-probability denominator: initial ratios are exactly one, but the numerator remains differentiable. A near-zero scalar loss can therefore have a nonzero gradient. Zero-variance groups supply no policy-gradient signal. Sampling and replay log probabilities are both saved to expose FP16/batch-shape differences.

With $V=r_s+r_p+r_d+r_c$ and $S$ the successful-STOP indicator:

$$R_A=S,$$
$$R_B=S+0.5\,\operatorname{clip}(\log(1+V_0)-\log(1+V_T),-1,1),$$
$$R_C=R_B-0.01\,N_{actions}-0.05\,N_{invalid}.$$

A/B have no extra invalid-action penalty, although invalid actions consume budget. Exact equality solves make stationarity and complementarity nearly zero; primal and dual feasibility provide most of the changing residual signal.

The curriculum uses 40 near-correct updates with horizon 4, 60 mixed-error updates with horizon 12, then 100 cold-start updates with horizon 20. Validation remains cold start. The three reward variants start independently from the same SFT adapter and use the same task-sampling seed.

## What the evidence supports

The reference experiment contains actual parameter updates, gradient audits, rollout logs and independent test certificates. It supports a gain over SFT for this model, budget and training seed. It does not establish multi-seed robustness, superiority to numerical solvers, or a decisive ranking of dense versus binary rewards. Syntax learning and decision learning are partly confounded in the zero-shot comparison. Nearly active constraints and normalized tolerance can permit different acceptable active sets.

The optional `train_grpo_trl.py` backend uses a tool-call dialect and requires separately trained tool-format SFT. It has not been validated end to end and is not a substitute for the main reproduction route.
