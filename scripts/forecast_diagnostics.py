"""Replay committed reaching actions to diagnose causal arrival-pose forecasts.

Future simulator states are evaluation targets only. Each predictor uses the
current/past measured positions and already committed future commands.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def evaluate(out):
    plans = sorted(out.glob('arrival_plan_task*.npz'))
    destination = out / 'forecast_diagnostics.json'
    previous = json.loads(destination.read_text(encoding='utf-8')) if destination.exists() else {'tasks': []}
    done = {r['task_id'] for r in previous['tasks']}
    pending = [p for p in plans if int(p.stem.removeprefix('arrival_plan_task')) not in done]
    if not pending:
        return
    os.environ['LIBERO_CONFIG_PATH'] = str(ROOT / '.runtime/libero_config')
    os.environ.setdefault('MUJOCO_GL','glfw' if os.name=='nt' else 'egl')
    sys.path.insert(0, str(ROOT / '.runtime/vendor/LIBERO-master'))
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()['libero_spatial']()
    for path in pending:
        task_id = int(path.stem.removeprefix('arrival_plan_task'))
        task = suite.get_task(task_id)
        with np.load(path, allow_pickle=False) as data:
            actions = data['committed'].copy()
        env = OffScreenRenderEnv(
            bddl_file_name=str(Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file),
            camera_heights=64, camera_widths=64)
        try:
            env.seed(0)
            env.reset()
            obs = env.set_init_state(suite.get_task_init_states(task_id)[0])
            for _ in range(10):
                obs, _, _, _ = env.step([0.] * 6 + [-1.])
            positions = [np.asarray(obs['robot0_eef_pos']).copy()]
            controller = env.env.robots[0].controller
            lower, upper = np.asarray(controller.input_min), np.asarray(controller.input_max)
            scale = (np.asarray(controller.output_max) - np.asarray(controller.output_min)) / (upper - lower)
            bias = (np.asarray(controller.output_max) + np.asarray(controller.output_min)) / 2
            deltas = (np.clip(actions[:, :6], lower, upper) - (upper + lower) / 2) * scale + bias
            for action in actions:
                obs, _, _, _ = env.step(action.tolist())
                positions.append(np.asarray(obs['robot0_eef_pos']).copy())
        finally:
            env.close()
        positions = np.asarray(positions)
        records = []
        lag = 4
        for t in range(4, len(actions) - lag + 1):
            current = positions[t]
            velocity = positions[t] - positions[t - 1]
            # Per-axis least-squares command gain, fitted strictly to the past.
            old_commands = deltas[t - 4:t, :3]
            old_moves = np.diff(positions[t - 4:t + 1], axis=0)
            gain = np.sum(old_commands * old_moves, axis=0) / (np.sum(old_commands ** 2, axis=0) + 1e-8)
            predictions = {
                'stale_pose': current,
                'controller_target_sum': current + deltas[t:t + lag, :3].sum(axis=0),
                'constant_velocity': current + lag * velocity,
                'past_calibrated_command': current + (deltas[t:t + lag, :3] * gain).sum(axis=0),
            }
            for name, prediction in predictions.items():
                records.append({'task_id': task_id, 'frame': t, 'lag': lag, 'method': name,
                                'position_error_mm': float(np.linalg.norm(prediction - positions[t + lag]) * 1000),
                                'is_selected_arrival': t == len(actions) - lag})
        previous['tasks'].append({'task_id': task_id, 'records': records})
        temporary = destination.with_suffix('.tmp.json')
        temporary.write_text(json.dumps(previous, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/research_pilot')
    evaluate(ROOT / parser.parse_args().output)
