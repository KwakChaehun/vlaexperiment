"""Measure physical endpoint agreement after the arrival-only intervention.

The reference is fresh-A's executed endpoint, not a ground-truth grasp pose.
These millimeter errors are teacher agreement, not task precision or success.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def rows(path):
    if not path.exists():return []
    result=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        try:result.append(json.loads(line))
        except json.JSONDecodeError:pass
    return result


def pending(out):
    old=rows(out/'arrival_endpoints.jsonl')
    completed={(r['variant'],r['task_id'],r['method']) for r in old}
    candidates=[]
    for variant,directory in [('target_sum',out),('past_calibrated_command',out/'arrival_calibrated')]:
        for r in rows(directory/'arrival_closed_loop.jsonl'):
            key=(variant,r['task_id'],r['method'])
            if r.get('status')=='completed' and r.get('first_actions') and key not in completed:
                candidates.append((variant,directory,r))
    return candidates


def evaluate(out):
    todo=pending(out)
    if not todo:return
    os.environ['LIBERO_CONFIG_PATH']=str(ROOT/'.runtime/libero_config')
    os.environ.setdefault('MUJOCO_GL','glfw' if os.name=='nt' else 'egl')
    sys.path.insert(0,str(ROOT/'.runtime/vendor/LIBERO-master'))
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite=benchmark.get_benchmark_dict()['libero_spatial']()
    for variant,directory,r in todo:
        task=suite.get_task(r['task_id'])
        with np.load(directory/f"arrival_plan_task{r['task_id']}.npz",allow_pickle=False) as d:
            committed=d['committed'].copy()
        env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=64,camera_widths=64)
        try:
            env.seed(0);env.reset();obs=env.set_init_state(suite.get_task_init_states(r['task_id'])[0])
            for _ in range(10):obs,_,_,_=env.step([0.]*6+[-1.])
            for action in committed:obs,_,_,_=env.step(action.tolist())
            initial=np.asarray(obs['robot0_eef_pos']).copy()
            for action in r['first_actions']:obs,_,_,_=env.step(action)
            endpoint={'position':np.asarray(obs['robot0_eef_pos']).tolist(),
                      'quaternion':np.asarray(obs['robot0_eef_quat']).tolist(),
                      'gripper_qpos':np.asarray(obs['robot0_gripper_qpos']).tolist()}
            record={'variant':variant,'task_id':r['task_id'],'method':r['method'],'endpoint':endpoint,
                    'motion_from_arrival_mm':float(np.linalg.norm(np.asarray(endpoint['position'])-initial)*1000),
                    'agreement_reference':'executed fresh A endpoint; not ground-truth task target'}
        finally:env.close()
        with (out/'arrival_endpoints.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(record,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/research_pilot')
    evaluate(ROOT/p.parse_args().output)
