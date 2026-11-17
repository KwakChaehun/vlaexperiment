"""Retiming the causal CPU forecaster on the same replayed pre-arrival inputs."""
import argparse
from collections import deque
import json
from pathlib import Path
import time
import numpy as np
import run_arrival_gr00t_probe as probe


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',default='results/arrival_gr00t_20261117');args=p.parse_args()
    root=Path(args.source);config=json.loads((root/'config.json').read_text())
    probe.base.setup_libero()
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import torch
    tasks={x['suite']:benchmark.get_benchmark_dict()[x['suite']]() for x in config['tasks']}
    diagnostics=probe.base.rows(root/'diagnostics.jsonl');cases={r['case']:r for r in diagnostics}
    result=[]
    for case,row in cases.items():
        spec=next(s for s in config['tasks'] if s['name']==row['task'])
        task=next(t for t in tasks[spec['suite']].tasks if t.name==spec['name'])
        data=np.load(root/row['observation_file'],allow_pickle=False)
        env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),
                              camera_heights=256,camera_widths=256,ignore_done=True)
        try:
            env.seed(0);env.reset()
            initial=torch.load(Path(get_libero_path('init_states'))/task.problem_folder/task.init_states_file,
                               weights_only=False,map_location='cpu')[row['initial_state']]
            obs=env.set_init_state(initial)
            for _ in range(10):obs,_,_,_=env.step([0.]*6+[-1.])
            look=config['lookahead_steps'];history=deque([probe.base.sim_obs(obs,task.language)],maxlen=look+1)
            for action in data['prefix'][:-look]:
                obs,_,_,_=env.step(action.tolist());history.append(probe.base.sim_obs(obs,task.language))
            history=list(history);prediction=probe.forecast(history,look)
            np.testing.assert_allclose(prediction.state,data['forecast_state'],atol=1e-7,rtol=0)
            np.testing.assert_array_equal(history[-1].images['image'],data['before_external'])
            samples=[]
            for _ in range(100):
                start=time.perf_counter();probe.forecast(history,look);samples.append((time.perf_counter()-start)*1000)
            result.append(dict(case=case,cpu_p50_ms=float(np.median(samples)),cpu_p95_ms=float(np.quantile(samples,.95)),
                repeats=100,input_replay_verified=True,scope='warm standalone CPU timing on original causal inputs; not concurrent measurement'))
            print(json.dumps(result[-1]),flush=True)
        finally:env.close()
    probe.base.save(root/'forecast_cpu_timings.json',result)


if __name__=='__main__':main()
