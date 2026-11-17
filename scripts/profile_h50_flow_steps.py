"""Measure the complete H50 expert and export every executed dense projection.

Native BF16, batch=1, N=1/4/10. Instrumented traces are separate from latency
measurements; the traces are not a measurement of GPU DRAM traffic.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import time

import run_gr00t_wrist_pilot as base
from vla_eval.gr00t16_arrival import Gr00t16Arrival


def stats(values):
    import numpy as np
    return dict(mean=statistics.mean(values), p50=float(np.median(values)),
                p95=float(np.percentile(values, 95)), samples=len(values))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--repeats', type=int, default=4)
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if (out/'measurements.jsonl').exists():
        raise ValueError('Use a new output directory')
    pilot = Gr00t16Arrival(str(base.ROOT/'checkpoints/gr00t_n16_libero'))
    t = pilot.torch
    cases, _ = base.cases(json.loads((base.ROOT/'configs/gr00t16_wrist_pilot.json').read_text()))
    def telemetry():
        return subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,memory.used,utilization.gpu', '--format=csv'], text=True)
    manifest = dict(H=50, N=[1,4,10], native_decoded_actions=pilot.output_horizon,
                    torch=t.__version__, cuda=t.version.cuda, gpu=t.cuda.get_device_name(),
                    before=telemetry(), dtype='bfloat16', batch=1,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with t.inference_mode():
        cond = pilot.condition(cases[0]['fresh'])
        for n in [1,4,10]:
            for _ in range(5):
                pilot.integrate(cond,50,n,37)
        pilot.synchronize()
        for repeat in range(args.repeats):
            for case in cases:
                ns = [1,4,10]; random.Random(917+repeat*100+len(base.rows(out/'measurements.jsonl'))).shuffle(ns)
                for n in ns:
                    pilot.synchronize(); before = time.perf_counter()
                    cond = pilot.condition(case['fresh'])
                    pilot.synchronize(); conditioned = time.perf_counter()
                    a,b = t.cuda.Event(enable_timing=True),t.cuda.Event(enable_timing=True)
                    a.record(); latent = pilot.integrate(cond,50,n,37); b.record()
                    pilot.synchronize(); finished = time.perf_counter()
                    decoded = pilot.decode(cond,latent); pilot.synchronize()
                    end = time.perf_counter()
                    assert list(latent.shape)==[1,50,128] and decoded.shape[0]==pilot.output_horizon
                    base.append(out/'measurements.jsonl',dict(case=case['id'], repeat=repeat,N=n,
                        backbone_tokens=cond[2].backbone_features.shape[1],
                        condition_ms=(conditioned-before)*1000, expert_wall_ms=(finished-conditioned)*1000,
                        expert_cuda_ms=a.elapsed_time(b),decode_ms=(end-finished)*1000,
                        total_ms=(end-before)*1000))
            rows = base.rows(out/'measurements.jsonl')
            print(json.dumps(dict(event='gpu_repeat', repeat=repeat+1,
                expert_mean_ms={n:statistics.mean(r['expert_wall_ms'] for r in rows if r['N']==n) for n in [1,4,10]})),flush=True)
        # Trace each distinct conditioning sequence length; no hooks in timed runs.
        modules = {name:mod for name,mod in pilot.head.named_modules()
                   if isinstance(mod,t.nn.Linear) or type(mod).__name__=='CategorySpecificLinear'}
        seen=set(); traces=[]
        for case in cases:
            cond=pilot.condition(case['fresh']); length=cond[2].backbone_features.shape[1]
            if length in seen: continue
            seen.add(length)
            for n in [1,4,10]:
                pending={}; events=[]; handles=[]
                for name,mod in modules.items():
                    def pre(mod, inputs, name=name):
                        x=inputs[0]; ev=t.cuda.Event(enable_timing=True); ev.record()
                        pending[name]=(ev,list(x.shape),x.numel()*x.element_size())
                    def post(mod, inputs, output, name=name):
                        ev=t.cuda.Event(enable_timing=True); ev.record()
                        start,shape,input_bytes=pending.pop(name)
                        events.append((dict(module=name, input_shape=shape, output_shape=list(output.shape),
                            vectors=output.numel()//output.shape[-1],input_dim=shape[-1],output_dim=output.shape[-1],
                            input_bytes=input_bytes,output_bytes=output.numel()*output.element_size(),
                            bias=mod.bias is not None if isinstance(mod,t.nn.Linear) else True),start,ev))
                    handles.extend([mod.register_forward_pre_hook(pre),mod.register_forward_hook(post)])
                a,b=t.cuda.Event(enable_timing=True),t.cuda.Event(enable_timing=True)
                a.record(); pilot.integrate(cond,50,n,37); b.record(); pilot.synchronize()
                for handle in handles:handle.remove()
                rr=[dict(row,cuda_ms=start.elapsed_time(end)) for row,start,end in events]
                assert len(rr)%n==0
                # Exact shape/order repetition is required before scaling simulated one-step work by N.
                shape_sequence=[(r['module'],r['vectors'],r['input_dim'],r['output_dim']) for r in rr]
                count=len(rr)//n
                assert shape_sequence==shape_sequence[:count]*n
                traces.append(dict(case=case['id'],backbone_tokens=length,N=n,
                    instrumented_expert_cuda_ms=a.elapsed_time(b),linear_cuda_sum_ms=sum(r['cuda_ms'] for r in rr),
                    operators=rr))
                print(json.dumps(dict(event='trace', tokens=length,N=n,linear_calls=len(rr))),flush=True)
        base.save(out/'traces.json',traces)
    rows=base.rows(out/'measurements.jsonl')
    summary=[dict(N=n,**{key:stats([r[key] for r in rows if r['N']==n]) for key in
        ['expert_wall_ms','expert_cuda_ms','condition_ms','decode_ms','total_ms']}) for n in [1,4,10]]
    manifest['after']=telemetry()
    base.save(out/'manifest.json',manifest); base.save(out/'gpu_summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
