"""RTX3090 measurements and explicitly hypothetical action-expert PIM scenarios.

Partition: preprocessing/backbone/state fusion remain on GPU/CPU. The complete
flow loop (action encoder, DiT, action decoder, latent updates) moves together.
Conditioning+initial latent enter once, final latent exits once. Weights stay on
the hypothetical PIM device. No PIM throughput, traffic or speedup is measured.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import run_gr00t_wrist_pilot as base
import numpy as np
from vla_eval.gr00t16_arrival import Gr00t16Arrival
from vla_eval.speculative_features import PredictedMLPCache


def tensor_bytes(x):return x.numel()*x.element_size()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'measurements.jsonl').exists():raise ValueError('Use a new output')
    pilot=Gr00t16Arrival(str(base.ROOT/'checkpoints/gr00t_n16_libero'));cache=PredictedMLPCache(pilot)
    cases,_=base.cases(json.loads((base.ROOT/'configs/gr00t16_wrist_pilot.json').read_text()))
    stages=['preprocess_ms','vision_ms','vlm_ms','state_and_fusion_ms','ae_ms','decode_ms','total_ms']
    for n in (1,4):
        for i in range(4):pilot.predict(cases[0]['fresh'],50,n,i)
    # Interleave modes, use multiple observations, and retain paired raw samples.
    for repeat in range(4):
        for index,case in enumerate(cases):
            modes=['fresh1','fresh4','predicted25_resume1']
            np.random.default_rng(1000*repeat+index).shuffle(modes)
            prediction=cache.capture(case['fresh'],37)
            for mode in modes:
                if mode.startswith('fresh'):
                    pilot.predict(case['fresh'],50,int(mode[-1]),37)
                    measured={k:pilot.stage[k] for k in stages}
                else:
                    # Exact predicted future is only a timing reference, not an accuracy test.
                    with pilot.torch.inference_mode():
                        cache.bank=prediction['mlps'];cache.ratio=.25;cache.stats=[];cache.mode='correct'
                        try:
                            pilot.synchronize();start=time.perf_counter()
                            cond=pilot.condition(case['fresh']);before=time.perf_counter()
                            latent=pilot.integrate(cond,50,4,37,start=3,latent=prediction['latent'])
                            pilot.synchronize();ae_ms=(time.perf_counter()-before)*1000;before=time.perf_counter()
                            pilot.decode(cond,latent);pilot.synchronize();decode_ms=(time.perf_counter()-before)*1000
                            total=(time.perf_counter()-start)*1000
                            measured={k:pilot.stage[k] for k in stages[:4]}
                            measured.update(ae_ms=ae_ms,decode_ms=decode_ms,total_ms=total)
                        finally:cache.mode='off';cache.bank=None
                base.append(out/'measurements.jsonl',dict(mode=mode,repeat=repeat,case=case['id'],**measured))
            del prediction
    rows=base.rows(out/'measurements.jsonl');summary=[]
    for mode in ['fresh1','fresh4','predicted25_resume1']:
        rr=[r for r in rows if r['mode']==mode]
        summary.append(dict(mode=mode,samples=len(rr),**{k:statistics.mean(r[k] for r in rr) for k in stages},
            total_p50_ms=float(np.median([r['total_ms'] for r in rr])),total_p95_ms=float(np.percentile([r['total_ms'] for r in rr],95))))
    base.save(out/'summary.json',summary)
    with pilot.torch.inference_mode():
        cond=pilot.condition(cases[0]['fresh']);noise=pilot.noise(cond,50,37)
        tensors={'backbone_features':cond[2].backbone_features,'state_features':cond[2].state_features,
                 'initial_latent':noise,'image_mask':cond[0].image_mask,
                 'attention_mask':cond[0].backbone_attention_mask,'embodiment_id':cond[1].embodiment_id}
        metadata={k:dict(shape=list(v.shape),dtype=str(v.dtype),bytes=tensor_bytes(v)) for k,v in tensors.items()}
        # Resolve variation in text token length across the twelve observations.
        payloads=[]
        for case in cases:
            c=pilot.condition(case['fresh'])
            inp=sum(tensor_bytes(v) for v in [c[2].backbone_features,c[2].state_features,noise,
                c[0].image_mask,c[0].backbone_attention_mask,c[1].embodiment_id])
            payloads.append(dict(case=case['id'],input_bytes=inp,output_bytes=tensor_bytes(noise)))
    modules={'action_encoder':pilot.head.action_encoder,'dit':pilot.head.model,
             'action_decoder':pilot.head.action_decoder,'position_embedding':pilot.head.position_embedding}
    weights={name:dict(parameters=sum(p.numel() for p in module.parameters()),
                       resident_bytes=sum(tensor_bytes(p) for p in module.parameters())) for name,module in modules.items()}
    base.save(out/'partition.json',dict(input_example=metadata,payloads=payloads,weights=weights,
        caveat='Resident parameter bytes, not measured DRAM traffic. Category-specific modules include unused embodiment categories. Weights assumed preloaded on PIM; no per-layer GPU round trips.'))
    # Separate instrumentation pass; CUDA event hooks do not contaminate table timings.
    events=[];handles=[];pending={}
    for name,module in modules.items():
        if name=='position_embedding':continue
        def before(mod,args,name=name):
            event=pilot.torch.cuda.Event(enable_timing=True);event.record();pending[name]=event
        def after(mod,args,output,name=name):
            event=pilot.torch.cuda.Event(enable_timing=True);event.record();events.append((name,pending.pop(name),event))
        handles.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])
    with pilot.torch.inference_mode():pilot.integrate(cond,50,4,37)
    pilot.synchronize()
    detail=[dict(module=name,cuda_ms=a.elapsed_time(b)) for name,a,b in events]
    for handle in handles:handle.remove()
    base.save(out/'expert_module_events.json',detail)
    # Hypothetical transfer uses the largest observed activation payload, not weights.
    transfer_bytes=max(p['input_bytes']+p['output_bytes'] for p in payloads)
    scenarios=[]
    for r in summary:
        rest=r['total_ms']-r['ae_ms']
        for k in [4,16]:
            scenarios.append(dict(mode=r['mode'],K=k,scenario='ideal_zero_expert_zero_transfer',
                predicted_total_ms=rest,speedup=r['total_ms']/rest,amortized_ms_per_action=rest/k,
                transfer_ms=0,expert_ms=0))
            for speed in [.5,1.,2.,4.]:
                for bandwidth in [8.,32.,128.]:
                    for fixed in [0.,.1,1.]:
                        transfer=transfer_bytes/(bandwidth*1e6)+fixed
                        total=rest+r['ae_ms']/speed+transfer
                        scenarios.append(dict(mode=r['mode'],K=k,scenario='hypothetical',expert_speedup=speed,
                            link_GB_s=bandwidth,fixed_roundtrip_ms=fixed,transfer_ms=transfer,
                            expert_ms=r['ae_ms']/speed,predicted_total_ms=total,
                            speedup=r['total_ms']/total,amortized_ms_per_action=total/k))
    base.save(out/'pim_scenarios.json',scenarios)
    base.save(out/'manifest.json',dict(gpu=pilot.torch.cuda.get_device_name(),H=50,
        model='same GR00T N1.6 LIBERO checkpoint as rollout experiments',scope=__doc__,
        timing='Stage wall clocks with CUDA synchronization; means add by construction; module event pass separate',
        source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        no_measured_pim=True,prediction_preparation_excluded_from_post_arrival_breakdown=True))
    print(json.dumps(dict(event='pim_profile_done',summary=summary,activation_roundtrip_bytes=transfer_bytes,weights=weights)),flush=True)
    cache.close()


if __name__=='__main__':main()
