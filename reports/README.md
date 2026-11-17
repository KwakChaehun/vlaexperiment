# Versioned result snapshots

- [Completed first CPU pilot](local_cpu_20261117/RESULTS.md): 24 offline observations, 27 arrival interventions, 36 H/K rollouts. All 63 live rollouts completed.
- [Follow-up snapshot](hk_matrix_20261117_snapshot/RESULTS.md): prior diagnostics at 10/12 observations; H/K rollouts had not started at export. This is not the live worker.

Each directory contains `snapshot.json` with export time and hashes. Ongoing local outputs remain under ignored `results/`; a push does not continuously sync running results. Models, input datasets, environments, GIFs and logs are omitted. See [research handoff](../docs/RESEARCH_HANDOFF.md) and [SSH GPU setup](../docs/SSH_GPU_RUNBOOK.md).
