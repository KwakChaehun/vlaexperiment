from pathlib import Path
import os, sys, json
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.runtime/vendor/LIBERO-master'))
os.environ['LIBERO_CONFIG_PATH'] = str(ROOT / '.runtime/libero_config')
os.environ.setdefault('MUJOCO_GL','glfw' if os.name=='nt' else 'egl')
import yaml
config_path = ROOT / '.runtime/libero_config'
config_path.mkdir(parents=True, exist_ok=True)
lib = ROOT / '.runtime/vendor/LIBERO-master/libero/libero'
(config_path / 'config.yaml').write_text(yaml.safe_dump({
    'benchmark_root':str(lib), 'bddl_files':str(lib/'bddl_files'),
    'init_states':str(lib/'init_files'), 'datasets':str(ROOT/'data'),
    'assets':str(lib/'assets')}))
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
suite = benchmark.get_benchmark_dict()['libero_spatial']()
task = suite.get_task(0)
env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256)
env.seed(0);env.reset()
obs = env.set_init_state(suite.get_task_init_states(0)[0])
for _ in range(10):obs,reward,done,info=env.step([0.]*6+[-1.])
print(json.dumps({'task':task.language,'image':list(obs['agentview_image'].shape),'eef':obs['robot0_eef_pos'].tolist(),'done':bool(done)}),flush=True)
env.close()
