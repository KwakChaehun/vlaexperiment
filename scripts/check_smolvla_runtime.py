"""Validate real sampler parity and actual short-H tensors on the selected device."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');args=parser.parse_args()
    from vla_eval.smolvla import SmolPilot
    from vla_eval.dataset import PairedDataset
    model=SmolPilot(ROOT/'checkpoints/smolvla_libero',ROOT/'checkpoints/smolvlm_config',device=args.device)
    dataset=PairedDataset(ROOT/'data/libero_pilot.npz');obs=dataset.observation(0);dataset.close()
    parity=model.sanity(obs)
    cond=model.condition(obs);actions,latent,ae=model.predict_condition(cond,4,10,0)
    result={**model.metadata,**parity,'short_action_shape':list(actions.shape),
            'actual_latent_shape':list(latent.shape),'latent_device':str(latent.device),
            'short_ae_ms':ae,'finite':bool(model.torch.isfinite(latent).all())}
    if not result['finite']:raise ValueError('Nonfinite action latent')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
