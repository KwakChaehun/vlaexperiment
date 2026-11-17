# Versioned result snapshots

- [RTX 3090 wrist reuse and H/K pilot](gpu_3090_wrist_20261117/RESULTS.md): 216 completed rollouts with GR00T N1.6 and SmolVLA, plus recorded-observation diagnostics.
- [Contact and containment H/K comparison](contact_hk_20261117/RESULTS.html): 180 matched contact-task rollouts and 54 plate-displacement trials. H50/K16, H50/K4, and H16/K4 achieved 53/60, 50/60, and 47/60 contact successes respectively; this does not establish a general ranking or tight-tolerance insertion performance.
- [GR00T precomputation and arrival correction](arrival_gr00t_20261117/RESULTS.html): 126 arrival-intervention rollouts and 21 terminal-segment diagnostics. Precomputation plus one fresh flow step did not outperform the fresh one-step baseline in this small probe. Oracle inputs and analytical overlap are explicitly labelled.
- [Completed first CPU pilot](local_cpu_20261117/RESULTS.md): 24 offline observations, 27 arrival interventions, 36 H/K rollouts. All 63 live rollouts completed.
- [Follow-up snapshot](hk_matrix_20261117_snapshot/RESULTS.md): prior diagnostics at 10/12 observations; H/K rollouts had not started at export. This is not the live worker.

Export metadata and hashes are in `snapshot.json` or `summary.json`, with model/code provenance in the included manifests. Ongoing local outputs remain under ignored `results/`; a push does not continuously sync running results. Models, input datasets, installed environments, GIFs and logs are omitted; environment package lists may be included. See [research handoff](../docs/RESEARCH_HANDOFF.md) and [SSH GPU setup](../docs/SSH_GPU_RUNBOOK.md).
