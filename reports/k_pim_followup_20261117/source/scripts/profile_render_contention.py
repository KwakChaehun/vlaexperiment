"""Stage breakdown while the same GPU renders a stationary simulator at 20Hz.

Latency diagnostic only: no success-rate claim, no PIM execution. All actions in
the render loop are zero motion. A single separate thread executes VLA requests.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import statistics
import time

import run_gr00t_wrist_pilot as base
import numpy as np
from vla_eval.gr00t16_arrival import Gr00t16Arrival
from vla_eval.speculative_features import PredictedMLPCache


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'measurements.jsonl').exists():raise ValueError('New output required')
    base.setup_libero()
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite=benchmark.get_benchmark_dict()['libero_goal']();task=next(t for t in suite.tasks if t.name=='put_the_cream_cheese_in_the_bowl')
    pilot=Gr00t16Arrival(str(base.ROOT/'checkpoints/gr00t_n16_libero'));cache=PredictedMLPCache(pilot)
    env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256,ignore_done=True)
    env.seed(0);env.reset()
    initial=pilot.torch.load(Path(get_libero_path('init_states'))/task.problem_folder/task.init_states_file,weights_only=False,map_location='cpu')[0]
    obs=env.set_init_state(initial)
    for _ in range(10):obs,_,_,_=env.step([0.]*6+[-1.])
    captured=cache.capture(base.sim_obs(obs,task.language),37)
    stages=['preprocess_ms','vision_ms','vlm_ms','state_and_fusion_ms','ae_ms','decode_ms','total_ms']
    def work(mode,observation):
        if mode.startswith('fresh'):
            pilot.predict(observation,50,int(mode[-1]),37)
            return {key:pilot.stage[key] for key in stages}
        with pilot.torch.inference_mode():
            cache.bank=captured['mlps'];cache.ratio=.25;cache.stats=[];cache.mode='correct'
            try:
                pilot.synchronize();start=time.perf_counter();cond=pilot.condition(observation);before=time.perf_counter()
                latent=pilot.integrate(cond,50,4,37,start=3,latent=captured['latent']);pilot.synchronize()
                ae=(time.perf_counter()-before)*1000;before=time.perf_counter();pilot.decode(cond,latent);pilot.synchronize()
                decode=(time.perf_counter()-before)*1000;total=(time.perf_counter()-start)*1000
                return {**{key:pilot.stage[key] for key in stages[:4]},'ae_ms':ae,'decode_ms':decode,'total_ms':total}
            finally:cache.mode='off';cache.bank=None
    modes=['fresh1','fresh2','fresh4','predicted25_resume1']
    lateness=[];result=[]
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            for mode in modes:pool.submit(work,mode,base.sim_obs(obs,task.language)).result()
            schedule=modes*16;np.random.default_rng(37).shuffle(schedule)
            tick=time.perf_counter()+.05
            for index,mode in enumerate(schedule):
                begin=time.perf_counter();future=pool.submit(work,mode,base.sim_obs(obs,task.language))
                # Continue real simulator/render ticks for at least 150 ms per request.
                while not future.done() or time.perf_counter()-begin<.15:
                    delay=tick-time.perf_counter()
                    if delay>0:time.sleep(delay)
                    lateness.append(max(0.,(time.perf_counter()-tick)*1000))
                    obs,_,_,_=env.step([0.]*6+[-1.]);tick+=.05
                row=dict(mode=mode,index=index,**future.result());base.append(out/'measurements.jsonl',row);result.append(row)
    finally:env.close();cache.close()
    summary=[dict(mode=mode,samples=sum(r['mode']==mode for r in result),**{key:statistics.mean(r[key] for r in result if r['mode']==mode) for key in stages}) for mode in modes]
    base.save(out/'summary.json',summary)
    base.save(out/'manifest.json',dict(scope=__doc__,control_hz=20,gpu=pilot.torch.cuda.get_device_name(),
        render_tick_lateness_p95_ms=float(np.percentile(lateness,95)),
        source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sparse_reference='Initial stationary frame cache; stage timing only, not a future-prediction accuracy result'))
    print(json.dumps(dict(summary=summary,render_p95_ms=float(np.percentile(lateness,95)))),flush=True)


if __name__=='__main__':main()
