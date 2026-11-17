"""Offline teacher dataset; simulator is not used by the learned predictor."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import run_gr00t_wrist_pilot as base
import numpy as np
import torch
from vla_eval.direct_feature_prediction import DirectFeaturePilot


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--output', required=True)
    args = p.parse_args(); cfg = json.loads(Path(args.config).read_text()); out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True); (out/'episodes').mkdir(exist_ok=True)
    if (out/'episodes.jsonl').exists(): raise ValueError('Use a new output directory')
    flat = sum(cfg['splits'].values(), [])
    assert len(flat) == len(set(flat)), 'Overlapping initial-state splits'
    base.save(out/'config.json', cfg); base.setup_libero()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suites = {s: benchmark.get_benchmark_dict()[s]() for s in {t['suite'] for t in cfg['tasks']}}
    pilot = DirectFeaturePilot(str(base.ROOT/cfg['checkpoint']))
    sources = ['scripts/collect_direct_features.py','vla_eval/direct_feature_prediction.py',
               'vla_eval/gr00t16_pilot.py','vla_eval/gr00t16_arrival.py']
    base.save(out/'manifest.json',dict(started_utc=base.now(), gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, source_hashes={s:hashlib.sha256((base.ROOT/s).read_bytes()).hexdigest() for s in sources},
        inputs='Origin/past observed features and state; known fresh-controller action prefix. Future features are labels only.',
        teacher='Frozen GR00T N1.6 H50/N4/K16; post-vlln feature target; current full simulator state never a predictor input'))
    completed = 0
    with torch.inference_mode():
        for split, inits in cfg['splits'].items():
            for init in inits:
                for task_index, spec in enumerate(cfg['tasks']):
                    begin = time.perf_counter(); task = next(t for t in suites[spec['suite']].tasks if t.name == spec['name'])
                    initial = torch.load(Path(get_libero_path('init_states'))/task.problem_folder/task.init_states_file,
                                         weights_only=False, map_location='cpu')[init]
                    env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),
                                             camera_heights=256, camera_widths=256, ignore_done=True)
                    try:
                        env.seed(0); env.reset(); obs = env.set_init_state(initial)
                        for _ in range(10): obs, _, _, _ = env.step([0.]*6+[-1.])
                        step = 0; history = []; frames = {}; pairs = []; previous = None; masks = None
                        seed = cfg['seed']+init*10000
                        def capture(at, raw):
                            view = base.sim_obs(raw, task.language); cond = pilot.condition(view)
                            frame = dict(step=at,state=view.state,images=view.images,
                                         feature=cond[2].backbone_features.detach().cpu(),
                                         sim_state=env.get_sim_state().copy())
                            return frame, cond
                        frame, cond = capture(0, obs)
                        while step < spec['start']:
                            actions = base.command_actions(pilot.decode(cond, pilot.integrate(cond, 50, 4, seed+step)))
                            previous = frame
                            for action in actions[:16]:
                                obs, _, _, _ = env.step(action.tolist()); history.append(action.copy()); step += 1
                            frame, cond = capture(step, obs)
                        already = bool(env.check_success())
                        frames[previous['step']] = previous; frames[step] = frame
                        masks = {k: getattr(cond[0], k).detach().cpu() for k in ['image_mask','backbone_attention_mask']}
                        # Validate bypass against native encoded conditioning at the SAME observation.
                        if completed == 0:
                            view = base.sim_obs(obs, task.language)
                            bypass_calls = [0]
                            handle = pilot.policy.model.backbone.register_forward_hook(lambda *a: bypass_calls.__setitem__(0,bypass_calls[0]+1))
                            bypass = pilot.condition_from_features(view, cond[2].backbone_features, masks)
                            handle.remove()
                            a = pilot.decode(cond, pilot.integrate(cond,50,4,37))
                            b = pilot.decode(bypass, pilot.integrate(bypass,50,4,37))
                            error = float(np.max(np.abs(a-b)))
                            assert error <= 1e-6 and bypass_calls[0] == 0
                            base.save(out/'parity.json',dict(bypass_action_max_abs=error,backbone_calls_in_bypass=bypass_calls[0]))
                        phase_success = already
                        while step < spec['end'] and not phase_success:
                            origin = step; prev_step = previous['step']
                            actions = base.command_actions(pilot.decode(cond, pilot.integrate(cond,50,4,seed+step)))
                            previous = frame
                            for offset, action in enumerate(actions[:16], 1):
                                applied = action.copy(); applied[:3] *= cfg['gain']
                                obs, _, ok, _ = env.step(applied.tolist()); history.append(applied.copy()); step += 1
                                phase_success |= bool(ok)
                                if offset in cfg['horizons']:
                                    frame, cond = capture(step,obs); frames[step] = frame
                                    for k in masks:
                                        assert torch.equal(masks[k],getattr(cond[0],k).cpu()), 'Token layout changed'
                                    pairs.append(dict(origin=origin,previous=prev_step,target=step,horizon=offset,
                                                      planned_actions=actions.copy(),success=bool(env.check_success())))
                        episode = dict(split=split,initial_state=init,task_index=task_index,task=task.name,
                                       instruction=task.language,spec=spec,seed=seed,masks=masks,frames=frames,pairs=pairs,
                                       applied_actions=np.asarray(history),already_successful_at_start=already,
                                       success=phase_success,goal_at_end=bool(env.check_success()))
                        filename=f'{split}_task{task_index}_init{init}.pt'
                        torch.save(episode,out/'episodes'/filename)
                        row = dict(file='episodes/'+filename,split=split,initial_state=init,task=task.name,
                                   pairs=len(pairs),frames=len(frames),success=phase_success,already_successful_at_start=already,
                                   sha256=hashlib.sha256((out/'episodes'/filename).read_bytes()).hexdigest(),wall_s=time.perf_counter()-begin)
                        base.append(out/'episodes.jsonl',row); completed += 1
                        base.save(out/'status.json',dict(status='running',completed=completed,updated_utc=base.now()))
                        print(json.dumps(row),flush=True)
                    finally:
                        env.close()
    base.save(out/'status.json',dict(status='completed',completed=completed,updated_utc=base.now()))


if __name__ == '__main__': main()
