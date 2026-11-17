"""Render recorded real-time applied actions, including actual zero-motion holds."""
import argparse
import json
from pathlib import Path
import shutil
import time

import run_gr00t_wrist_pilot as base
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from vla_eval.gr00t16_arrival import Gr00t16Arrival


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--viewer',required=True)
    args=p.parse_args();src=Path(args.input);viewer=Path(args.viewer);viewer.mkdir(parents=True,exist_ok=True)
    out=src/'replays';out.mkdir(exist_ok=True);config=json.loads((src/'config.json').read_text());rows=base.rows(src/'rollouts.jsonl')
    methods=['fresh4_k16','fresh1_k16','whole_s25_k16','whole_gate_k16']
    groups={}
    for r in rows:groups.setdefault((r['task'],r['initial_state']),{})[r['method']]=r
    # Show a jointly successful case with the greatest measured hold reduction.
    candidates=[k for k,g in groups.items() if all(g[m]['success'] for m in methods)]
    if not candidates:candidates=list(groups)
    case=max(candidates,key=lambda k:groups[k]['fresh4_k16']['hold_steps']-groups[k]['whole_s25_k16']['hold_steps'])
    rr=groups[case];spec=next(t for t in config['tasks'] if t['name']==case[0]);seed=config['seed']+case[1]*10000
    base.setup_libero()
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite=benchmark.get_benchmark_dict()[spec['suite']]();task=next(t for t in suite.tasks if t.name==case[0])
    pilot=Gr00t16Arrival(str(base.ROOT/config['checkpoint']))
    initial=pilot.torch.load(Path(get_libero_path('init_states'))/task.problem_folder/task.init_states_file,weights_only=False,map_location='cpu')[case[1]]
    def make():
        env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256,ignore_done=True)
        env.seed(0);env.reset();obs=env.set_init_state(initial)
        for _ in range(10):obs,_,_,_=env.step([0.]*6+[-1.])
        return env,obs
    env,obs=make();prefix=[]
    while len(prefix)<spec['start']:
        chunk=base.command_actions(pilot.predict(base.sim_obs(obs,task.language),50,4,seed+len(prefix)))
        for a in chunk[:16]:
            obs,_,_,_=env.step(a.tolist());prefix.append(a.copy())
            if len(prefix)>=spec['start']:break
    env.close();streams={};verification=[]
    font=ImageFont.load_default()
    def frame(obs,method,physical_step):
        data=base.sim_obs(obs,task.language);canvas=Image.new('RGB',(256,572),'#111827');d=ImageDraw.Draw(canvas)
        d.text((7,5),method,fill='white',font=font)
        d.text((7,22),f't={physical_step/20:.2f}s  wait ticks={rr[method]["hold_steps"]}',fill='white',font=font)
        label='SUCCESS' if physical_step>=rr[method]['physical_steps'] and rr[method]['success'] else 'running'
        d.text((7,39),label,fill='#86efac',font=font)
        canvas.paste(Image.fromarray(data.images['image']),(0,60));canvas.paste(Image.fromarray(data.images['wrist_image']),(0,316))
        return canvas
    for method in methods:
        env,obs=make()
        for a in prefix:obs,_,_,_=env.step(a.tolist())
        frames=[frame(obs,method,0)];mismatches=0
        for index,entry in enumerate(rr[method]['trace']):
            obs,_,ok,_=env.step(entry['action']);mismatches+=int(bool(ok)!=entry['goal'])
            if (index+1)%2==0 or index+1==len(rr[method]['trace']):frames.append(frame(obs,method,index+1))
        endpoint=base.sim_obs(obs,task.language).state
        error=float(np.max(np.abs(endpoint-np.array(rr[method]['endpoint_state']))))
        if mismatches or error>1e-6:raise ValueError(f'Recorded trace differs from replay: {method}, {error}, {mismatches}')
        verification.append(dict(method=method,endpoint_max_abs=error,goal_trace_mismatches=mismatches,success=rr[method]['success']))
        env.close();streams[method]=frames
    combined=[]
    for i in range(max(map(len,streams.values()))):
        canvas=Image.new('RGB',(1024,572))
        for j,m in enumerate(methods):canvas.paste(streams[m][min(i,len(streams[m])-1)],(j*256,0))
        combined.append(canvas)
    combined[0].save(out/'comparison.gif',save_all=True,append_images=combined[1:],duration=100,loop=0)
    # Data are re-rendered without VLA inference; applied hold actions are retained.
    base.save(out/'verification.json',dict(case=case,selection='Joint-success case with largest fresh4-to-whole25 hold-count reduction; illustrative, not random',
        frame_stride=2,simulation_hz=20,preparation_waits_omitted_from_video=True,actual_control_hold_actions_included=True,trials=verification,
        phase_timings={m:{k:rr[m][k] for k in ['phase_wall_s','startup_s','hold_steps','success','stable_success']} for m in methods}))
    shutil.copy2(out/'comparison.gif',viewer/'realtime_comparison.gif')
    page=f'''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>실시간 제어 비교</title><style>body{{font:17px/1.6 system-ui;background:#111827;color:#e5e7eb;margin:24px}}img{{max-width:100%}}a{{color:#93c5fd}}td,th{{padding:8px;border:1px solid #4b5563}}table{{border-collapse:collapse}}</style><h1>20Hz 실제 제어 기록 재생</h1><p>{task.language} / 초기 상태 {case[1]}</p><p>각 열 위: 외부 카메라, 아래: 손목 카메라. 실제 제어에서 실행한 대기 동작까지 재생했습니다. 시작 전 미래 bank 준비 시간은 영상에서 생략했으며 아래 표에 표시합니다. 모두 성공한 사례 중 대기 감소가 큰 사례를 골랐으므로 평균을 대표하지 않습니다.</p><img src="realtime_comparison.gif"><table><tr><th>방식</th><th>phase 시간</th><th>준비 시간</th><th>대기 tick</th><th>0.5초 목표 유지</th></tr>'''
    for m in methods:
        r=rr[m];page+=f'<tr><td>{m}</td><td>{r["phase_wall_s"]:.2f}s</td><td>{r["startup_s"]:.2f}s</td><td>{r["hold_steps"]}</td><td>{r["stable_success"]}</td></tr>'
    page+='</table><p><a href="report/RESULTS.html">전체 실험·PIM 보고서</a> · <a href="/">이전 영상</a></p>'
    (viewer/'index.html').write_text(page)
    print(json.dumps(dict(case=case,verification=verification)),flush=True)


if __name__=='__main__':main()
