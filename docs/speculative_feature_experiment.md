# Prediction-relative feature correction on RTX 3090

The question is whether features computed for a **predicted future observation**
can be corrected cheaply when the real observation arrives. This differs from
reusing a previous real image or shortening action horizon H.

## First implementation

`vla_eval/speculative_features.py` captures each vision/VLM MLP's input and output
at a predicted future frame. At arrival, it ranks token discrepancies against that
prediction at each layer, evaluates the MLP only on the selected rows, and combines
them with the predicted outputs. Attention, patch embedding, and state processing
are fully recomputed. This is a partial sparse implementation, not sparse attention
and not a reproduction of an existing paper. Fixed budgets 25% and 50% do not assume
the residual actually has that sparsity; the resulting success loss tests that choice.

Related mechanism: [Eventful Transformers, ICCV 2023](https://wisionlab.com/project/eventful-transformers/)
uses gates/buffers and sparse attention updates for temporal reuse. Our reference
cache is the predicted future, and this pilot only sparsifies MLP computation.

The future bank also holds the action expert latent after 3 of 4 integration steps.
At arrival we update features, insert the actual robot state, and execute the last
native integration step. This untrained correction is not guaranteed to recover
the fully recomputed four-step output when the prediction is wrong.

## Forecast model and controlled error

A separate nominal LIBERO simulator plus the same frozen VLA recursively predicts
future actions, images and states. It starts from a replay of **past** real commands.
Full initial simulator state and exact dynamics are privileged: this is an optimistic
world-model diagnostic, not a learned/deployable visual predictor. No actual future
images, reference future trajectories, or future success annotations are supplied.

After the common prefix, actual execution scales translational commands by 0.98;
the forecast uses nominal gain 1. Rotation/gripper commands are unchanged. Prediction
error also accumulates because corrected actual actions differ from the original
forecast plan. Forecast banks remain immutable; long-bank later states are not
silently replaced with true future observations.

## Comparisons

All methods use H=50, K=4, same checkpoint, same per-control-step initial noise.
Each bank frame is predicted by full N=4 inference with its first four actions used
to advance the nominal world. Actual sparse correction resumes the final N=4 step.

* fresh4 / fresh1: fresh observations, no prediction.
* early16_s25: predict the first 16 control steps once; then fresh4 for the remainder.
* rolling16_s25: repeatedly predict the next 16 steps from the current real state.
* whole_s25 / whole_s50: predict the entire remaining evaluation window once.
* whole_dense: same long bank and partial action latent, full feature recomputation.
* whole_replay: execute nominal predicted actions without correction.

Windows fixed before outcomes: cream cheese in bowl steps 64–192; top drawer/bowl
128–256; bottom drawer/bowl/close 160–352. At 20Hz, the long windows are 6.4/9.6s;
the short window is 0.8s. These cover substantial remaining manipulation, but are
not automatically detected precision phases or a dedicated insertion benchmark.
Initial states 0 and 4 on all three tasks; report all six, including failures.

Stop at the first successful 4-action chunk or the fixed end. Report both any native
success within the chunk and goal at chunk end. There is no separate rescue controller
except the explicitly specified fresh4 remainder in the early-only arm.

## Timing and validity

Report prediction preparation wall time, model-only preparation time, post-arrival
latency, cache memory, and actual sequential planning+correction service cost.
Prediction preparation includes past-history replay and future rendering, which are
diagnostic simulator costs, not predictions of a learned predictor's runtime.
The simulator pauses while inference runs. There is no real asynchronous/control
deadline experiment, no measured PIM, and no claim that preparation can be hidden.
CPU token selection, gathers/scatters, GPU synchronization and input processing are
inside post-arrival timings. Offline diagnostics are outside that timing.

GPU checks: native versus split sampler identity, 100% update identity, immutable
forecast cache after correction. Unit tests verify that partial update really sends
fewer rows to the MLP, targets the prediction residual, and rejects shape mismatches.
Same-input partial updates may have small BF16 differences because GEMM shape changes.

This screen cannot establish success-rate superiority from six paired cases, or
exclude stronger sparse attention/predictor/correction designs when it fails.
