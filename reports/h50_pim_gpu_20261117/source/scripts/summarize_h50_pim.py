"""Account for a serialized GPU/PIM projection split; never label it hardware timing."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import statistics


def physical_key(op):
    return (op['vectors'],(op['output_dim']+4095)//4096*4096,
            (op['input_dim']+255)//256*256)


def account(trace, shapes, conversion, remainder_ms, bandwidth_GBs, roundtrip_us):
    if bandwidth_GBs<=0 or roundtrip_us<0:raise ValueError('Invalid link assumptions')
    total=Counter();groups=defaultdict(Counter)
    for op in trace['operators']:
        key=physical_key(op);sim=shapes[key]
        assert key==(sim['vectors'],sim['padded_output_dim'],sim['padded_input_dim'])
        values=dict(pim_compute_ms=sim['compute_ms'],pim_readback_ms=sim['readback_ms'],
            gpu_pack_reduce_ms=conversion[(op['vectors'],op['input_dim'],op['output_dim'])],
            link_bytes=sim['padded_input_bytes']+sim['partial_output_bytes'],linear_calls=1)
        total.update(values)
        name=op['module']
        if '.attn1.to_k' in name or '.attn1.to_v' in name:
            group='cross_KV' if op['input_dim']==2048 else 'self_KV'
        elif '.ff.' in name:group='FFN'
        elif '.norm1.linear' in name:group='AdaNorm_projection'
        else:group='other_projections'
        groups[group].update(values)
    total['gpu_remainder_ms']=remainder_ms
    total['link_transfer_ms']=total['link_bytes']/(bandwidth_GBs*1e6)
    total['link_fixed_ms']=total['linear_calls']*roundtrip_us/1000
    total['hybrid_expert_estimate_ms']=sum(total[k] for k in [
        'pim_compute_ms','pim_readback_ms','gpu_pack_reduce_ms','gpu_remainder_ms','link_transfer_ms','link_fixed_ms'])
    return dict(total),dict(groups)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True)
    args=p.parse_args();root=Path(args.directory)
    def read(name):return json.loads((root/name).read_text())
    traces=read('gpu/traces.json');gpu=read('gpu/gpu_summary.json')
    samples=[json.loads(x) for x in (root/'gpu/measurements.jsonl').read_text().splitlines()]
    pim=read('pim/summary.json');rem=read('remainder/summary.json')
    shapes={(r['vectors'],r['padded_output_dim'],r['padded_input_dim']):r for r in pim['cases']}
    conversion={(r['vectors'],r['input_dim'],r['output_dim']):r['cuda_ms']['p50'] for r in rem['packing_reduction']}
    remainder={(r['backbone_tokens'],r['N']):r['wall_ms']['p50'] for r in rem['remainder']}
    rows=[];group_rows=[]
    for tr in traces:
        for bw in [8.,32.,128.]:
            for fixed in [0.,10.,100.]:
                values,groups=account(tr,shapes,conversion,remainder[(tr['backbone_tokens'],tr['N'])],bw,fixed)
                rows.append(dict(N=tr['N'],backbone_tokens=tr['backbone_tokens'],bandwidth_GBs=bw,roundtrip_us=fixed,**values))
                if bw==32 and fixed==0:
                    group_rows.append(dict(N=tr['N'],backbone_tokens=tr['backbone_tokens'],groups=groups))
    summary=[]
    for n in [1,4,10]:
        reference=next(r for r in gpu if r['N']==n)
        weight=Counter(r['backbone_tokens'] for r in samples if r['N']==n)
        count=sum(weight.values())
        for bw in [8.,32.,128.]:
            for fixed in [0.,10.,100.]:
                rr=[r for r in rows if r['N']==n and r['bandwidth_GBs']==bw and r['roundtrip_us']==fixed]
                fields=set(rr[0])-{'N','backbone_tokens','bandwidth_GBs','roundtrip_us'}
                values={k:sum(r[k]*weight[r['backbone_tokens']] for r in rr)/count for k in fields}
                nonexpert=statistics.median(r['condition_ms']+r['decode_ms'] for r in samples if r['N']==n)
                summary.append(dict(N=n,bandwidth_GBs=bw,roundtrip_us=fixed,
                    gpu_expert_p50_ms=reference['expert_wall_ms']['p50'],
                    gpu_vla_p50_ms=reference['total_ms']['p50'],
                    **values,hybrid_vla_estimate_ms=values['hybrid_expert_estimate_ms']+nonexpert))
    report=dict(summary=summary,by_length=rows,groups=group_rows,
        method='Native uninstrumented GPU p50 vs operator-sum serialized PIM mapping + cached-output GPU replay + packing/reduction + assumed link. No functioning full PIM backend.',
        link='Effective GB/s, no overlap; fixed roundtrip charged for EACH linear call. Weights resident; no preload or whole-model placement simulation.',
        readback='16 FP16 lane partial sums per padded output, all transferred; GPU reduction/cast/bias measured separately.',
        extrapolation='N1/4/10 graph order and counts traced; each unique FULL BATCH shape cycle-simulated once, then summed by occurrence. No single-GEMV x 50 shortcut.',
        exclusions=['PIM model numerical accuracy','whole-model physical placement and relocation','CPU PIM driver and command issue cost','DMA buffer management','GPU/PIM contention','thermal/power/resource matching'],
        no_pim_hardware_measurement=True)
    (root/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    with (root/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    selected=[r for r in summary if r['bandwidth_GBs']==32 and r['roundtrip_us']==0]
    print(json.dumps(selected,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10})
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    x=np.arange(3)
    for offset,key,label,color in [(-.2,'gpu_expert_p50_ms','RTX 3090 measured','#2878b5'),(.2,'hybrid_expert_estimate_ms','GPU + serial-GEMV PIM estimate','#d96932')]:
        vals=[r[key] for r in selected]
        bars=axs[0].bar(x+offset,vals,.38,label=label,color=color)
        axs[0].bar_label(bars,fmt='%.1f',padding=3)
    axs[0].set(xticks=x,xticklabels=['N=1','N=4','N=10'],ylabel='Expert latency (ms)',title='H=50; 32 GB/s assumed link, zero fixed latency')
    axs[0].legend(fontsize=8)
    row=selected[0]
    keys=['pim_compute_ms','pim_readback_ms','gpu_remainder_ms','gpu_pack_reduce_ms','link_transfer_ms']
    vals=[row[k] for k in keys]
    axs[1].barh(['PIM matrix','PIM readback','GPU remainder','GPU pack/reduce','Assumed link'],vals,color=['#d96932','#dbad80','#2878b5','#6db5d9','#888888'])
    axs[1].invert_yaxis();axs[1].set(xlabel='Latency contribution (ms)',title='N=1 offload estimate breakdown')
    fig.suptitle('Serialized mapping estimate, not full PIM execution or hardware measurement',fontsize=11)
    fig.tight_layout();fig.savefig(root/'latency.png',dpi=170);fig.savefig(root/'latency.svg');plt.close(fig)
    svg=root/'latency.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')


if __name__=='__main__':main()
