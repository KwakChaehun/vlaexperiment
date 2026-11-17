"""GPU implementation checks and latency screening before closed-loop evaluation."""
import argparse
import json
import time
from pathlib import Path

import run_gr00t_wrist_pilot as base
import numpy as np
from vla_eval.gr00t16_arrival import Gr00t16Arrival
from vla_eval.speculative_features import PredictedMLPCache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    pilot = Gr00t16Arrival(str(base.ROOT/'checkpoints/gr00t_n16_libero'))
    cache = PredictedMLPCache(pilot)
    cases, std = base.cases(json.loads((base.ROOT/'configs/gr00t16_wrist_pilot.json').read_text()))
    obs = cases[0]['fresh']
    parity = pilot.split_parity(obs)
    prediction = cache.capture(obs, 37)
    expected, _ = pilot.finish(obs, 37)
    exact, _, _ = cache.correct(obs, prediction, 37, 1)
    error = float(np.max(np.abs(expected-exact)))
    if error > 1e-6:
        raise ValueError(f'All-token correction not identical to native: {error}')
    before = {n: (a.clone(), b.clone()) for n, (a, b) in prediction['mlps'].items()}
    cache.correct(cases[0]['stale'][4], prediction, 37, .25)
    if any(not pilot.torch.equal(a, before[n][0]) or not pilot.torch.equal(b, before[n][1])
           for n, (a, b) in prediction['mlps'].items()):
        raise ValueError('Correction modified immutable predicted cache')
    del before
    base.save(out/'parity.json', dict(parity=parity, all_token_action_max_abs=error,
        cache_immutable=True, mlp_names=list(cache.originals),
        cache_bytes_per_frame=cache.bytes(prediction),
        scope='Sparse MLP only; attention and patch embeddings fully recomputed; same-observation oracle test'))
    methods=['fresh4','fresh1','sparse0','sparse25','sparse50','sparse75','sparse100']
    results=[]
    for repeat in range(9):
        order=methods.copy();np.random.default_rng(repeat).shuffle(order)
        for method in order:
            if method.startswith('fresh'):
                actions, ms=pilot.finish(obs,37,steps=int(method[-1]));groups={}
            else:
                actions, ms, groups=cache.correct(obs,prediction,37,int(method[6:])/100)
            row=dict(repeat=repeat, method=method, post_ms=ms,
                action_max_abs=float(np.max(np.abs(actions-expected))),groups=groups)
            results.append(row);base.append(out/'timings.jsonl',row)
    summary=[dict(method=m,p50_ms=float(np.median([r['post_ms'] for r in results if r['method']==m and r['repeat']>0])),
        max_action_error=max(r['action_max_abs'] for r in results if r['method']==m)) for m in methods]
    base.save(out/'summary.json',summary)
    print(json.dumps(dict(event='feature_probe_done',summary=summary,
                         cache_MiB=cache.bytes(prediction)/2**20)),flush=True)
    cache.close()


if __name__=='__main__':main()
