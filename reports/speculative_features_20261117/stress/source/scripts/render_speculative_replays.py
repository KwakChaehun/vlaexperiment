"""Re-run selected matched trials, verify outcomes, and render actual camera GIFs."""
import argparse
import json
from pathlib import Path
import shutil

import run_gr00t_wrist_pilot as base
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from vla_eval.gr00t16_arrival import Gr00t16Arrival
from vla_eval.speculative_features import PredictedMLPCache


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--viewer',required=True)
    args=p.parse_args();root=Path(args.input);viewer=Path(args.viewer);out=root/'replays';out.mkdir(exist_ok=True)
    config=json.loads((root/'config.json').read_text());rows=base.rows(root/'rollouts.jsonl')
    methods=['fresh4','rolling16_s25','whole_s25']
    cases={r['case'] for r in rows};groups={c:{r['method']:r for r in rows if r['case']==c} for c in cases}
    # Select a visible disagreement if present, otherwise earliest cream-cheese case.
    def score(c):
        g=groups[c];values=[g[m]['success'] for m in methods]
        return (len(set(values)), 'cream_cheese' in c, -g[methods[0]]['initial_state'])
    case=max(sorted(cases),key=score);reference=groups[case];rr=reference[methods[0]]
    spec=next(t for t in config['tasks'] if t['name']==rr['task']);seed=config['seed']+rr['initial_state']*10000
    base.setup_libero()
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite=benchmark.get_benchmark_dict()[spec['suite']]();task=next(t for t in suite.tasks if t.name==spec['name'])
    pilot=Gr00t16Arrival(str(base.ROOT/config['checkpoint']));cache=PredictedMLPCache(pilot)
    initial=pilot.torch.load(Path(get_libero_path('init_states'))/task.problem_folder/task.init_states_file,weights_only=False,map_location='cpu')[rr['initial_state']]
    def make():
        env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256,ignore_done=True)
        env.seed(0);env.reset();obs=env.set_init_state(initial)
        for _ in range(10):obs,_,_,_=env.step([0.]*6+[-1.])
        return env,obs
    env,obs=make();prefix=[]
    while len(prefix)<spec['start']:
        actions=base.command_actions(pilot.predict(base.sim_obs(obs,task.language),50,4,seed+len(prefix)))
        for action in actions[:16]:
            obs,_,_,_=env.step(action.tolist());prefix.append(action.copy())
            if len(prefix)>=spec['start']:break
    anchor=env.get_sim_state().copy();env.close()
    streams={};verification=[]
    font=ImageFont.truetype('/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf',14) if Path('/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf').exists() else ImageFont.load_default()
    def frame(obs,method,step,success):
        fresh=base.sim_obs(obs,task.language)
        canvas=Image.new('RGB',(256,552),'#111827');d=ImageDraw.Draw(canvas)
        d.text((8,4),method,fill='white',font=font)
        d.text((8,23),f'step {step} / '+('SUCCESS' if success else 'running'),fill='#86efac' if success else 'white',font=font)
        canvas.paste(Image.fromarray(fresh.images['image']),(0,40))
        canvas.paste(Image.fromarray(fresh.images['wrist_image']),(0,296))
        return canvas
    for method in methods:
        env,obs=make();history=[a.copy() for a in prefix]
        for action in prefix:obs,_,_,_=env.step(action.tolist())
        if np.max(np.abs(env.get_sim_state()-anchor))>1e-7:raise ValueError('Replay prefix mismatch')
        step=spec['start'];success=False;bank={};frames=[frame(obs,method,step,False)]
        while step<spec['end'] and not success:
            if method!='fresh4' and (step==spec['start'] or (method=='rolling16_s25' and step not in bank)):
                bank.clear();future,fobs=make()
                try:
                    for a in history:fobs,_,_,_=future.step(a.tolist())
                    if np.max(np.abs(future.get_sim_state()-env.get_sim_state()))>1e-7:raise ValueError('Forecast clone mismatch')
                    stop=min(step+(16 if method=='rolling16_s25' else spec['end']-step),spec['end'])
                    for t in range(step,stop,4):
                        pred=cache.capture(base.sim_obs(fobs,task.language),seed+t);bank[t]=pred
                        for a in base.command_actions(pred['actions'])[:4]:fobs,_,_,_=future.step(a.tolist())
                finally:
                    future.close();env.sim._render_context_offscreen.gl_ctx.make_current()
            fresh=base.sim_obs(obs,task.language)
            if method=='fresh4':actions,_=pilot.finish(fresh,seed+step)
            else:actions,_,_=cache.correct(fresh,bank[step],seed+step,.25)
            for a in base.command_actions(actions)[:min(4,spec['end']-step)]:
                applied=a.copy();applied[:3]*=rr['gain'];obs,_,ok,_=env.step(applied.tolist())
                history.append(applied.copy());step+=1;success|=bool(ok)
            frames.append(frame(obs,method,step,success))
            bank.pop(step-4,None)
        end=base.sim_obs(obs,task.language).state
        original=reference[method]
        error=float(np.max(np.abs(end-np.array(original['endpoint_state']))))
        matched=success==original['success'] and step==original['steps'] and error<1e-6
        verification.append(dict(method=method,case=case,success=success,steps=step,
            original_success=original['success'],original_steps=original['steps'],endpoint_max_abs=error,matched=matched))
        bank.clear();env.close();streams[method]=frames
        frames[0].save(out/(method+'.gif'),save_all=True,append_images=frames[1:],duration=200,loop=0)
        print(json.dumps(verification[-1]),flush=True)
    combined=[]
    for i in range(max(map(len,streams.values()))):
        canvas=Image.new('RGB',(256*3,552))
        for j,method in enumerate(methods):canvas.paste(streams[method][min(i,len(streams[method])-1)],(j*256,0))
        combined.append(canvas)
    combined[0].save(out/'comparison.gif',save_all=True,append_images=combined[1:],duration=200,loop=0)
    base.save(out/'verification.json',dict(case=case,selection='Illustrative outcome disagreement where available; not randomly sampled',
        fps=5,control_hz=20,frame_stride=4,all_matched=all(r['matched'] for r in verification),trials=verification,
        timing='Simulation time only; GPU prediction/arrival waits omitted; first successful frame held in comparison'))
    for path in out.glob('*.gif'):shutil.copy2(path,viewer/('feature_'+path.name))
    label=' / '.join(f"{r['method']}: {'성공' if r['success'] else '실패'}" for r in verification)
    (viewer/'feature.html').write_text(f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>미래 feature 보정 비교</title><style>body{{font:17px/1.6 system-ui;background:#111827;color:#e5e7eb;margin:24px}}img{{max-width:100%;height:auto}}a{{color:#93c5fd}}</style><h1>미래 feature 보정 비교</h1><p>{task.language} · 초기 상태 {rr['initial_state']} · 이동 gain {rr['gain']}</p><p>{label}</p><p>왼쪽: fresh 4단계 / 가운데: 짧게 반복 예측 + 25% 보정 / 오른쪽: 남은 구간 전체 예측 + 25% 보정</p><p>각 열 위는 외부 카메라, 아래는 손목 카메라입니다. GIF는 시뮬레이션 시간으로 재생하며 준비·추론 대기는 생략합니다. 성공 후 화면은 고정됩니다. 전체 성공률을 대표하는 사례는 아닙니다.</p><img src="feature_comparison.gif"><p>기존 평가와 재생 실행 결과 일치: {all(r['matched'] for r in verification)}</p><p><a href="/">기존 H/K 영상</a></p></html>''')
    cache.close()


if __name__=='__main__':main()
