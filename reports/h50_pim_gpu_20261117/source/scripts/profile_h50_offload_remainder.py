"""Measure GPU work retained by the serialized projection-offload scenario.

Replay exact cached linear outputs to time the remaining dependency graph.
This is a counterfactual timing probe, NOT a functioning PIM backend. Separately
measure FP16 input packing and GPU reduction of 16 partial sums per padded
output. Link latency is not measured. Accuracy is only checked for replay.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import run_gr00t_wrist_pilot as base
from profile_h50_flow_steps import stats
from vla_eval.gr00t16_arrival import Gr00t16Arrival


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--trace',required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise ValueError('Use new output')
    pilot=Gr00t16Arrival(str(base.ROOT/'checkpoints/gr00t_n16_libero'));t=pilot.torch
    cases,_=base.cases(json.loads((base.ROOT/'configs/gr00t16_wrist_pilot.json').read_text()))
    modules={name:m for name,m in pilot.head.named_modules()
             if isinstance(m,t.nn.Linear) or type(m).__name__=='CategorySpecificLinear'}
    original={name:m.forward for name,m in modules.items()}
    traces=json.loads(Path(args.trace).read_text());records=[]
    with t.inference_mode():
        seen=set()
        for case in cases:
            cond=pilot.condition(case['fresh']);length=cond[2].backbone_features.shape[1]
            if length in seen:continue
            seen.add(length)
            for n in [1,4,10]:
                cached=defaultdict(list)
                for name,m in modules.items():
                    def capture(*a,_name=name,**kw):
                        result=original[_name](*a,**kw);cached[_name].append(result);return result
                    m.forward=capture
                reference=pilot.integrate(cond,50,n,37)
                positions=defaultdict(int)
                for name,m in modules.items():
                    def replay(*a,_name=name,**kw):
                        pos=positions[_name];positions[_name]+=1;return cached[_name][pos]
                    m.forward=replay
                def execute():
                    positions.clear();return pilot.integrate(cond,50,n,37)
                try:
                    actual=execute();assert t.equal(actual,reference)
                    for _ in range(5):execute()
                    measurements=[]
                    for _ in range(24):
                        pilot.synchronize();before=time.perf_counter()
                        a,b=t.cuda.Event(enable_timing=True),t.cuda.Event(enable_timing=True)
                        a.record();execute();b.record();pilot.synchronize()
                        measurements.append(dict(wall_ms=(time.perf_counter()-before)*1000,cuda_ms=a.elapsed_time(b)))
                    row=dict(backbone_tokens=length,N=n,exact_replay_parity=True,
                        cached_linear_outputs=sum(len(v) for v in cached.values()),
                        wall_ms=stats([r['wall_ms'] for r in measurements]),
                        cuda_ms=stats([r['cuda_ms'] for r in measurements]),raw=measurements)
                    records.append(row);print(json.dumps({k:v for k,v in row.items() if k!='raw'}),flush=True)
                finally:
                    for name,m in modules.items():m.forward=original[name]
        # Streaming conversion and reduction kernels, isolated from graph replay.
        shapes=sorted({(r['vectors'],r['input_dim'],r['output_dim']) for tr in traces for r in tr['operators']})
        conversion=[]
        for vectors,indim,outdim in shapes:
            pin=(indim+255)//256*256;pout=(outdim+4095)//4096*4096
            source=t.zeros((vectors,indim),device='cuda',dtype=t.bfloat16)
            packed=t.zeros((vectors,pin),device='cuda',dtype=t.float16)
            partial=t.zeros((vectors,pout,16),device='cuda',dtype=t.float16)
            bias=t.zeros(outdim,device='cuda',dtype=t.bfloat16)
            def convert():
                packed[:,:indim].copy_(source)
                # Padded result discarded, useful outputs reduce on GPU.
                return partial[:,:outdim,:].sum(-1,dtype=t.float32).to(t.bfloat16)+bias
            for _ in range(10):convert()
            pairs=[]
            for _ in range(64):
                a,b=t.cuda.Event(enable_timing=True),t.cuda.Event(enable_timing=True)
                a.record();convert();b.record();pairs.append((a,b))
            pilot.synchronize();ms=[a.elapsed_time(b) for a,b in pairs]
            conversion.append(dict(vectors=vectors,input_dim=indim,output_dim=outdim,
                cuda_ms=stats(ms),raw_cuda_ms=ms))
    base.save(out/'summary.json',dict(scope=__doc__,remainder=records,packing_reduction=conversion,
        caveat='Optimistic isolated GPU work; no real PIM, DMA/driver stalls, allocation for received outputs, or accuracy equivalence. Packed padding assumed pre-zeroed; weights resident.'))


if __name__=='__main__':main()
