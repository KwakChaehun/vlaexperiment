"""Native N1.6 integration split without changing the frozen pilot adapter."""
import time
import numpy as np
from .gr00t16_pilot import Gr00t16Pilot


class Gr00t16Arrival(Gr00t16Pilot):
    def noise(self,condition,horizon,seed):
        t=self.torch
        return t.randn((1,self.native_horizon,self.head.action_dim),
            generator=t.Generator(device=self.head.device).manual_seed(seed),
            device=self.head.device,dtype=condition[2].backbone_features.dtype)[:,:horizon].clone()

    def integrate(self,condition,horizon,steps,seed,start=0,stop=None,latent=None):
        stop=steps if stop is None else stop
        if not 0<=start<=stop<=steps or (start>0 and latent is None):raise ValueError('Invalid split')
        t,head=self.torch,self.head
        backbone,ai,features,_=condition
        x=self.noise(condition,horizon,seed) if latent is None else latent.clone()
        if x.shape!=(1,horizon,head.action_dim):raise ValueError('Incorrect latent shape')
        for i in range(start,stop):
            ticks=t.full((1,),int(i/float(steps)*head.num_timestep_buckets),device=head.device)
            act=head.action_encoder(x,ticks,ai.embodiment_id)
            if head.config.add_pos_embed:
                act=act+head.position_embedding(t.arange(horizon,device=head.device))[None]
            args=dict(hidden_states=t.cat([features.state_features,act],dim=1),
                encoder_hidden_states=features.backbone_features,timestep=ticks)
            if head.config.use_alternate_vl_dit:
                args.update(image_mask=backbone.image_mask,backbone_attention_mask=backbone.backbone_attention_mask)
            velocity=head.action_decoder(head.model(**args),ai.embodiment_id)[:,-horizon:]
            x=x+velocity/float(steps)
        return x

    def decode(self,condition,latent):
        values=self.policy.processor.decode_action(latent.float().cpu().numpy(),
                    self.policy.embodiment_tag,condition[3])
        return np.concatenate([values[k] for k in self.policy.modality_configs['action'].modality_keys],-1)[0]

    def prepare_prior(self,obs,seed):
        with self.torch.inference_mode():
            self.synchronize();start=time.perf_counter()
            cond=self.condition(obs)
            part=self.integrate(cond,50,4,seed,stop=3)
            self.synchronize();partial_ms=(time.perf_counter()-start)*1000
            full=self.integrate(cond,50,4,seed,start=3,latent=part)
            actions=self.decode(cond,full)
            self.synchronize();full_ms=(time.perf_counter()-start)*1000
        return dict(partial=part,full=full,actions=actions,partial_ms=partial_ms,full_ms=full_ms)

    def finish(self,obs,seed,steps=4,latent=None,start_step=0):
        with self.torch.inference_mode():
            self.synchronize();start=time.perf_counter()
            cond=self.condition(obs)
            x=self.integrate(cond,50,steps,seed,start=start_step,latent=latent)
            actions=self.decode(cond,x)
            self.synchronize();elapsed=(time.perf_counter()-start)*1000
        return actions,elapsed

    def split_parity(self,obs,seed=37):
        reference=self.predict(obs,50,4,seed)
        with self.torch.inference_mode():
            cond=self.condition(obs)
            full=self.integrate(cond,50,4,seed)
            part=self.integrate(cond,50,4,seed,stop=3)
            split=self.integrate(cond,50,4,seed,start=3,latent=part)
            direct=self.decode(cond,full)
            maximum=float((full-split).abs().max().item())
        error=float(np.max(np.abs(reference-direct)))
        if maximum!=0 or error>1e-6:raise ValueError(f'Split sampler mismatch {maximum}, {error}')
        return dict(split_latent_max_abs=maximum,existing_sampler_action_max_abs=error)
