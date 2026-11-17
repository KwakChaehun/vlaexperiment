"""Load the pinned real SmolVLA checkpoint and execute one CPU inference."""
from pathlib import Path
import sys, time, json
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".runtime/data-deps"))
sys.path.insert(0, str(ROOT / ".runtime/pilot-deps"))
import torch
from safetensors.torch import load_file
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from transformers import AutoTokenizer
import numpy as np
from lerobot.configs.types import PolicyFeature, FeatureType

torch.set_num_threads(4)
start = time.perf_counter()
raw = json.loads((ROOT / "checkpoints/smolvla_libero/config.json").read_text())
raw.pop('type')
for key in ('input_features','output_features'):
    raw[key] = {k:PolicyFeature(type=FeatureType(v['type']),shape=tuple(v['shape'])) for k,v in raw[key].items()}
cfg = SmolVLAConfig(**raw)
cfg.device = "cpu"
cfg.load_vlm_weights = False
cfg.vlm_model_name = str(ROOT / "checkpoints/smolvlm_config")
print("constructing model", flush=True)
policy = SmolVLAPolicy.from_pretrained(str(ROOT / "checkpoints/smolvla_libero"), config=cfg)
policy.eval()
print("model loaded", time.perf_counter()-start, flush=True)
stats = load_file(str(ROOT / "checkpoints/smolvla_libero/policy_preprocessor_step_5_normalizer_processor.safetensors"))
tokenizer = AutoTokenizer.from_pretrained(cfg.vlm_model_name)
with np.load(ROOT / "data/libero_subset.npz", allow_pickle=False) as d:
    state = torch.from_numpy(d['state'][0].copy()).float()
    state = (state - stats['observation.state.mean'])/(stats['observation.state.std'] + 1e-8)
    batch = {'observation.state':state[None]}
    for key, source in zip(cfg.image_features, ['image__image','image__wrist_image']):
        batch[key] = torch.from_numpy(d[source][0,0].copy()).permute(2,0,1)[None].float()/255
    language = tokenizer(str(d['instruction'][0])+'\n',return_tensors='pt',padding='longest',max_length=48,truncation=True)
    batch['observation.language.tokens'] = language['input_ids']
    batch['observation.language.attention_mask'] = language['attention_mask'].bool()
with torch.inference_mode():
    t=time.perf_counter()
    actions=policy.predict_action_chunk(batch)
    print(json.dumps({'action_shape':list(actions.shape),'seconds':time.perf_counter()-t,
                      'finite':bool(actions.isfinite().all()),'normalized_first_action':actions[0,0].tolist()}),flush=True)
