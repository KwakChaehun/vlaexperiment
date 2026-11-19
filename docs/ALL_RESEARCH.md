# VLA·추론 가속 연구 통합 기록

갱신: 2026-11-19 KST. 이 파일은 지금까지의 질문, 구현, 실측, 실패 및 해석 수정을 한곳에 모은 기록입니다. 숫자는 각 실험의 표본·판정·측정 경계에 종속됩니다. 서로 다른 실험의 성공 횟수나 시간은 합치지 않습니다. 개별 원자료를 따라갈 수 있도록 상대 링크를 유지합니다.

## 1. 연구 목표와 현재 결론

초기 목표는 손목 영상의 빠른 변화와 접촉·삽입 정밀도를 고려하면서 VLA 계산 지연을 줄이는 것이었습니다. 이후 미래 특징 예측, sparse correction, PIM offload, 여러 모델의 검증, 여러 noise 후보 선택, GPU·RTL 최적화, 촉각 제어 및 스트리밍 행동 생성으로 범위를 넓혔습니다.

현재 확인한 것:

- H16/K4가 H50/K16보다 정확하다는 일반적 근거를 얻지 못했습니다. 자주 재계획해도 성공률이 오르지 않는 경우가 있습니다.
- 미래 특징 예측은 오래된 특징의 단순 재사용보다 나아질 수 있지만, 강한 fresh 저반복 기준선을 이기지 못했습니다.
- 현재 Samsung PIMSimulator의 토큰 순차 GEMV 매핑은 GPU보다 느립니다. PIM 전체의 불가능성을 뜻하지 않습니다. 현재 신규 연구에서 PIM은 제외합니다.
- π가 GR00T 후보를 검증하는 방식의 추가 이득은 작고, 일부 초기 이득은 종료 판정 수정 후 사라졌습니다.
- 여러 noise 후보 합의는 일부 장면을 구하지만 해치는 장면도 있습니다. 저렴한 단일 후보 대비 확실한 우위는 아직 없습니다.
- 같은 관측의 KV 재사용, CUDA Graph, 반복 전처리 제거 등 소프트웨어 이득은 실제 확인했습니다. 이 자체는 새 하드웨어 기여가 아닙니다.
- RTL의 일부 FFN 타일에서는 cycle 절약을 확인했지만 전체 VLA 속도·전력·면적 이득을 구현한 것은 아닙니다.
- Tabero의 빠른 신경망 재계획이 접촉 성공률을 높인다는 근거는 아직 없습니다.
- FlashVLA는 공개 모델로 실행 가능했고 8회 파일럿 모두 목표 판정을 통과했습니다. 다른 모델 대비 우위나 정밀 삽입 성능은 검증하지 않았습니다.

## 2. 용어·장비·평가에서 혼동하면 안 되는 점

|항목|정의|
|---|---|
|H|생성하는 action latent의 길이. 자기회귀 언어 토큰을 H번 순차 생성한다는 뜻이 아님|
|K|다시 관측·계획하기 전 실제 실행하는 행동 수|
|N|flow/denoising 반복 횟수. K와 다름|
|fresh|현재 관측을 넣어 해당 설정으로 새 추론. 정답·oracle·최고 정확도라는 뜻이 아님|
|GR00T H50|현재 LIBERO 체크포인트에서는 H50 latent를 계산하고 공식 decoder가 16개 물리 명령을 반환|
|π0.5 H10|사용한 체크포인트의 학습 설정. H16/50 변경은 재학습 없는 추론 길이 확장|
|FlashVLA|공개 체크포인트는 4슬롯×10행동=40행동 버퍼, 매 호출 한 단계 갱신|
|장비|RTX3090 24GB, 호스트 RAM 약15GiB, Rocky Linux9. 최신 GPU 배율을 사양으로 환산하지 않음|
|주요 실행|PyTorch2.9/CUDA12.8, π/Tabero는 별도 JAX 환경. 모델별 버전은 각 보고서에 기록|
|주간 사용량|과거 사용자가 확인한 표시값3%→5%는 계정 전체 반올림 값. 도구로 현재 계정 사용량을 조회한 값이 아님|

대부분의 closed-loop 실험은 추론 중 시뮬레이터가 멈춥니다. 실제 GPU 지연 동안 로봇이 계속 움직이는 실시간 평가와 구분합니다. 예외로 시간 지연을 명시적으로 넣은 과거 실험은 별도 기록합니다. LIBERO의 놓기·서랍·접시 작업은 sub-mm 삽입 성공률 검증이 아닙니다. 학습 task와 겹칠 수 있는 공개 체크포인트의 새로운 reset을 새로운 held-out task라고 부르지 않습니다.

## 3. 초기 H/K·손목 갱신 비교

[초기 216회 보고서](../reports/gpu_3090_wrist_20261117/RESULTS.md). GR00T N1.7은 필수 backbone 접근 승인 문제로 사용하지 못했고 공개 N1.6을 사용했습니다. SmolVLA도 초기 비교 후 주 연구에서는 제외했습니다.

초기 12개 조건에서 GR00T H50/K16, H50/K4, H16/K4는 모두12/12 성공, SmolVLA H16/K4/N2는9/12였습니다. 천장 효과가 강하며 모델 학습 예산도 통제하지 않았습니다. H50→16은 호출 시간을 약0.6ms만 줄였습니다. K4는 재계획 횟수가 증가해 행동당 계산 비용이 커집니다.

접촉·포함 조건을 늘린 [추가 234회](../reports/contact_hk_20261117/RESULTS.html): 접촉180회와 접시 이동54회입니다.

|설정|접촉 목표 통과|유지 판정|호출 중앙값|행동당 계산|
|---|---:|---:|---:|---:|
|H50/K16|53/60|43/60|65.51ms|4.34ms|
|H50/K4|50/60|42/60|65.18ms|16.41ms|
|H16/K4|47/60|41/60|64.70ms|16.27ms|

작업별 예외는 있으나 H16/K4의 일반적 우위를 지지하지 않습니다. 일부 작업의 작은 차이로 사용 조건을 확정할 수 없습니다. 외부 카메라를 오래된 입력으로 바꾸는 초기6조건에서는 fresh6/6→4/6이었고, 손목 stale은5/6이었습니다. 외부 영상도 중요하다는 관찰이며 카메라별 일반 순위는 아닙니다.

## 4. 미래 관측·특징 선행 계산과 sparse correction

[도착 보정](../reports/arrival_gr00t_20261117/RESULTS.html):126개 도착 개입과21개 종단 진단. 선행 계산 뒤 fresh flow 한 번을 추가해도 fresh 한 번 기준선을 넘지 못했습니다.

[예측 관측 sparse correction](../reports/speculative_features_20261117/RESULTS.html): nominal simulator로 미래 이미지를 만드는 특권적 예측기 사용. 2% 명령 불일치48회,10%30회,exact-future3회. MLP 일부 갱신으로 도착 후35.9ms 대 fresh1 41.1ms까지 줄였지만, 예측 준비에 모델·simulator 반복 실행이 필요했습니다. 2%에서는 모든 방식5/6,10%에서는 fresh4 4/6, fresh1·짧은/전체 준비·무보정 재생5/6이었습니다. 보정의 독립 성공률 이득은 미확인입니다. Attention은 dense로 남습니다.

[K16·실시간 비교](../reports/k_pim_followup_20261117/RESULTS.html): 동기식72회,20Hz 지연 반영96회,반복48회. fresh4/K16과 whole25/K16은8/12로 같았지만, 준비 비용까지 포함하면 함께 성공한 경우 whole 쪽이 평균3.364초 느렸습니다.

[직접 특징 예측 초기](../reports/direct_features_20261117/RESULTS.md): 학습18/검증6/시험12 에피소드,offline165쌍,동기식144회. 첫 fresh chunk가 즉시 도착 시점 실패를 가렸고, 첫 도착부터 적용하면 direct5/12,corrected7/12,fresh4 10/12였습니다.

[접근 단계까지 포함한 후속](../reports/direct_arrival_features_20261117/RESULTS.md):49k-parameter predictor,offline249쌍,72회. 새 시험 조건에서 fresh4 9/12,fresh2 11/12,stale4/12,direct8/12,corrected7/12. 준비·갱신 포함 행동당 direct3.377ms 대 fresh2 3.366ms였습니다. N이 많을수록 정확하다는 가정은 성립하지 않습니다. 미래 예측 전체의 불가능성을 의미하지는 않습니다.

## 5. PIM: 실제 simulator와 GPU 비교

[공식 GEMV·shape 기능 검사](pim_simulation.md), [GPU projection 비교](../reports/pim_gpu_comparison_20261117/RESULTS.md), [H50/N별 전체 분해](../reports/h50_pim_gpu_20261117/RESULTS.md).

Samsung PIMSimulator를 실제 빌드·실행했습니다. 공식4096출력 검사 통과. full-batch driver 주소 계산을 수정하고24,576개 합성 출력도 검증했습니다. GPU projection은48shape/layout/dtype/vector 조건을 반복 측정했습니다.

|H50,N|GPU 전체 expert 실측|PIM 행렬 합계|직렬 GPU+PIM expert 추정|
|---|---:|---:|---:|
|1|8.06ms|116.30ms|202.41ms|
|4|31.16ms|465.18ms|809.16ms|
|10|76.99ms|1162.96ms|2023.05ms|

PIM 열은 simulator cycles×tCK 및 실제 trace별 연산 합계입니다. 호스트에서 simulator를 실행하는 데 걸린 시간이 아닙니다. GPU+PIM 열은 GPU 잔여·부분합·전송을 더한 조건부 계산이며 실제 통합 PIM 시스템 실행이 아닙니다.32GB/s 링크·가중치 사전 상주 등 가정이 있습니다. 토큰별 GEMV, padding, 외부 reduction 비용이 컸습니다. GPU는 여러 action token을 GEMM으로 처리합니다. PIM 보드·전력·면적 비교를 수행하지 않았습니다.

## 6. 큰 모델 후보 선택·교차 검증

[GR00T/π0.5/OpenVLA-OFT](../reports/large_vla_candidates_20261118/RESULTS.md):31상태×186branch,첫8행동만 개입 후 동일 정책으로 이어감. 유지 성공 GR16/31,π19/31,OFT16/31,합의·그룹CV 검증18/31,정답을 아는 oracle21/31. Oracle은 배포 가능한 방식이 아닙니다.

[1% 불일치 재계산](../reports/pi05_gr00t_disagreement_20261118/RESULTS.md):6공유 출발 상태,rolling66회+seed 대조24회.459검사가 전부 재계산 전후에도 reject였습니다. 검증기 없이 같은 retry seed를 쓰면36개 gated trajectory의 전체 행동·최종 물리가 정확히 같았습니다. 이 설정에서 검증 이득은 없었습니다.

[GR 후보를 π로 선택](../reports/gr00t_pi05_research_20261118/RESULTS.md):1101branch(학습165 포함),평가 출발 상태60개를 여러 방식으로 재사용. 새36조건 K16 유지 성공 GR21,GR3합의27,π선택29. 그러나 πN2 단독/K8도29/36,행동당15.02ms로 이중 모델18.44ms보다 저렴했습니다. GR expert batch127→84ms는 유지27→24/36으로 바뀌었고, π padding 제거246→206ms는24/24행동 trace를 보존했습니다.

[선택적 검증](../reports/selective_verification_20261118/RESULTS.html):새24후반 상태,240개 branch physics replay 감사. K16 유지 GR합의18,항상π19,선택적π18,무작위π20/24. 선택적 모델은97판단 중9회만π를 호출했으나 합의보다 추가 성공을 만들지 못했습니다.

### 중요한 해석 수정

[조작 시점별 재평가](../reports/verification_moments_20261118/RESULTS.html):24trajectory의97개 상관된 시점,291논리 비교/257새 물리 실행. 초기에 목표가 한 번 참이 되는 즉시 정지하여 GR의 뒤쪽 gripper-open 행동이 잘린 사례가 있었습니다. 같은 committed chunk와 동일 settling을 적용한 뒤 첫 출발 상태 기준 GR23/24,π개입22/24,조기재계획21/24였습니다. 전체97시점에서 π rescue0,harm5는 모두 하나의 trajectory에 집중됐습니다. 초기 cream-cheese rescue를 확정적 근거로 쓰면 안 됩니다.

또 float64 실제 명령을 float32로 바꾼 작은 차이도 궤적을 바꾸는 것을 발견했습니다. 해당 예비 실행은 제외하고 원본 명령으로 재실행했습니다. 수정된 결과를 우선합니다.

## 7. 여러 noise의 GR00T·GPU 가속·RTL

[36후반 상태/612회](../reports/multinoise_acceleration_20261118/RESULTS.html):무작위 단일 후보26/36,순차3·anchor4·batch8/16 모두27/36. CUDA Graph 순차3은132.22→104.59ms이며36개 후보풀·명령·최종 상태 모두 일치했습니다. Anchor4는88.69ms지만 후보 집합이 다릅니다. 후보를 늘리는 이득이 저렴한 random-single 대비 작았습니다.

같은 관측의 cross-attention K/V 재사용은 이미 batch된 호출77.40→69.89ms,12/12 cached-batch action parity였습니다. 이것은 관측이 바뀌기 전의 정확한 재사용입니다.

[확장1072회/144새 reset](../reports/pices_pair_research_20261118/RESULTS.html):모든 물리 trace와15,830개 선택/K 판단을 감사했습니다. 같은96조건 비교:

|방식|성공|전체 호출 p50|
|---|---:|---:|
|단일 N1|86/96|40.32ms|
|단일 N4|81/96|56.96ms|
|4후보 medoid N2|88/96|62.92ms|
|4후보 medoid N4|84/96|86.85ms|

N2 medoid 대 N1은 rescue9/harm7,순증2/96,task-stratified bootstrap95%[-6.25,+10.42]percentage points,McNemar p=.8036. 마지막 별도48reset은41/42/45/41로 단일N1의 이전 우위도 일관되게 재현되지 않았습니다. Fresh reset은 학습 밖 task를 뜻하지 않습니다.

- 모든 graph가 상주한 무작위 interleaved840회,전체 여러 ablation2400timing.
- Candidate-shared KV는 BF16 32회+pair8 32회에서 pool/score/index/K/command/finalstate를 정확히 보존.
- Fused INT8는 GPU85.30→71.48ms였지만 양자화 품질이 달라집니다.
- BF16 nominal+INT8 residual은 느리고 성공도 감소하여 현재 구현을 기각했습니다.
- 실제 Verilator RTL:같은 sparse core 기준 FFN-up K1536×16출력 타일.실제 activation384개,정수출력13,160,448개+합성1792개 모두 일치.
- Cycle 절약은 독립 후보 약8.3%,nominal/probe 쌍 약43%. 전체 VLA43%가 아닙니다. 주파수·면적·DRAM stall·정량화·비선형·전력·전체 시스템을 포함하지 않습니다.
- 최적화된 전체 호출71.6ms 중 대표 FFN GEMM은9.31ms. 타일 단위 큰 이득을 전체 속도로 대입하지 않습니다.

A3의 전체 검증 알고리즘을 단순 medoid와 동일시하면 안 됩니다. PICES의 nominal/probe 민감도 및 K 선택도 별도입니다. A3/PICES/SpecVLA/Deltoris/Zora 등과 겹치는 일반적 consensus·delta·shared KV·mixed precision을 새 아이디어로 주장하지 않습니다.

## 8. 촉각·힘 피드백 Tabero

[전체 보고서](../reports/tactile_latency_20261119/RESULTS.html). 공개 π0 force-prefix 및 force-suffix 체크포인트를 사용했습니다. 서로 다른 학습 모델이므로 구조 하나의 인과 비교가 아닙니다. 후자는 같은 영상·언어 KV를 유지하면서 힘·state를 갱신할 수 있습니다.

데이터는 NathanWu7/tabero TRAIN split 중11작업군의 강한/부드러운 잡기22episode,668시점입니다. 균등 시점+힘 변화가 큰 시점을 섞어 접촉 전환을 과대표집했습니다. 입력 변형8684건,시범 동작 비교2004건입니다. 출력RMSE는 실제 로봇 종단 오차나 성공률이 아닙니다.

|N|fresh 전체 호출|같은 영상 KV 사용|정확한 출력 일치|
|---|---:|---:|---:|
|1|107.35ms|9.30ms|130/130|
|2|111.25ms|13.53ms|130/130|
|4|120.36ms|22.38ms|130/130|
|10|145.43ms|47.74ms|130/130|

같은 입력520/520쌍 일치입니다. 오래된 영상의 품질 보장은 아닙니다. 다른 별도 쌍별 실험에서 전처리 재사용은432/432 일치,N10 46.35→44.74ms. LoRA 병합까지 적용하면N4 21.04→17.51,N10 46.20→39.75ms이나 비트 일치0쌍,접촉 품질 보존 미검증입니다.

Nsight/HLO에서 큰 concatenate는 KV가 아니라 반복 결합되는 MLP base/LoRA 가중치였습니다. 병합 후 N10 해당 커널 합은13.58→0.47ms. GPU커널 합집합 약29.4ms가 남았습니다. 프로파일러 시간과 일반 timing을 섞지 않습니다. 병합 trace의 첫 native 호출은 cold여서 steady 비교에서 제외 표시했습니다.

스케줄링60회:20Hz 요청,영상 목표5Hz. 개선한 N4는720/720마감 준수와 영상4.917Hz를 동시에 달성,p99응답약19ms,영상나이389–393ms. N10은 요청 마감을 지켜도 영상0.25–1.08Hz까지 떨어지고 나이가 초 단위였습니다. 이 구현의 한계이며 모든 GPU scheduler의 불가능성 증명은 아닙니다.

Isaac5 문제로 별도6.2 환경을 만들고 physics namespace,자동단위변환,손가락 joint 순서를 보정했습니다. 원본 설치는 유지했습니다. 기록행동 실제 물리 재생으로 환경을 확인했습니다. 원논문 환경의 정확한 재현이라고 주장하지 않습니다.

|작은 수프→바구니 파일럿|목표 통과|
|---|---:|
|prefix N10/K10|3/3|
|prefix N4/K10|최초2/3,반복1/3|
|prefix N1/K10|1/3|
|prefix N4/K4 또는K1|각0/3|
|suffix N4/K10|1/3|
|suffix N4/K1|0/3|

지연·캐시 조건 포함39회,서로 다른 초기 상태는3개입니다. 공개 위치·접촉 조건8연속프레임을 사용하며 명시적 release나 삽입을 뜻하지 않습니다. RPC동안 물리가 멈춥니다. K1에서 grasp 실패가 관찰됐으나 원인을 noise 하나로 확정하지 않습니다. 기록 TRAIN 동작 오차에서는 오히려 suffix N4가 N10보다 작은 경우가 많았습니다. 다른 checkpoint의 offline/closed-loop 결과를 하나의 인과 비교로 묶지 않습니다.

## 9. FlashVLA 및 Cosmos Policy

FlashVLA(CoRL2026): 서로 다른 noise 단계의 행동 청크를 계속 갱신하는 rolling buffer. 공개 π0.5 LIBERO 모델,4×10설정을 유지했습니다. 임시 FP32 중복 할당을 피하는 strict loading으로 GPU8.7–8.9GB에서 로드했습니다. 공개 tokenizer를 정확히 변환·검사했습니다.

|구현,준비된 GPU 입력64회씩|p50|p99|
|---|---:|---:|
|eager|83.54ms|84.23ms|
|공식 fusion+compile|62.60ms|64.19ms|
|SDPA 교체|63.95ms|65.47ms|
|SDPA K/V view 확장|63.48ms|65.09ms|

영상 처리·VLM·expert·CPU출력 복사 포함,CPU tokenizer/normalizer와 simulator 제외. Gaussian 고정 버퍼는 timing·수치 비교용이며 실제 streaming 버퍼 분포를 대표하지 않습니다. 공식 compile도 eager와 비트 일치하지 않으며 SDPA도 수치 차이가 있습니다.

Compiled만 실제4작업×2초기 상태 실행:크림치즈→그릇,위서랍→그릇,아래서랍→그릇/닫기,그릇→접시 모두2/2. 초기 state/두영상 모두 원본과 일치. 첫3대기chunk도 제한시간에 포함. Native goal 기준,GR의 release/hold 강화 판정과 다릅니다. 첫 실행8.83초/6.18초 stall을 보존했고 원인을 compiler trace로 분리하지 않았습니다. 전체148호출p99 3316ms,첫 trial 전체 제외136호출p50 60.94/p99 65.26ms. 초기 비용을 숨기지 않습니다.

Cosmos Policy(ICLR2026)는 공식 소스만 분석했습니다. 체크포인트 실행·3090 시간은 아직 없습니다. ALOHA 행동8×미래3×가치5 검색은 논문에서8H100/4.9초,어려운 두 실물 작업의 평균 부분완료 점수+12.5점을 보고합니다. 성공률+12.5%p로 바꾸어 쓰면 안 됩니다. 코드에는 latent 직접전달·VAE encode 생략이 이미 있습니다. batch1순차ensemble,서로 다른 반복수,단계별가중치 상주가 조사 대상입니다. 먼저 강한 GPU batching 및 선택 품질 재현이 필요합니다.

## 10. 선행연구와 차별성 판단

|연구|이미 다루는 부분|우리에게 남는 확인|
|---|---|---|
|[A3](https://arxiv.org/html/2605.11567v4)|후보·검증 및 저정밀 verifier|단순 합의를 새 방법으로 주장하지 않을 것|
|[PICES](https://arxiv.org/html/2610.19656)|noise perturbation,민감도·실행 길이|동일 알고리즘의 구체적 dataflow 이득|
|[SpecVLA](https://arxiv.org/html/2608.15636)|선택적 검증·알고리즘/구조 공동 설계|상태 기반 호출·residual 자체의 중복|
|[KerColle,PACT2026](https://arxiv.org/html/2609.22335)|VLM/action head CTA 간섭과 하드웨어 동시 스케줄링|일반적 overlap만으로는 차별성이 약함|
|[AT-VLA](https://arxiv.org/html/2605.07308)|느린 시각과 빠른 촉각|두 흐름 분리 자체는 새롭지 않음|
|[Tabero,ICML2026](https://proceedings.mlr.press/v306/wu26cb.html)|힘·위치 hybrid 제어|기존 저수준 힘 제어 위에서 신경망 가속의 추가 효과|
|[SEAM](https://arxiv.org/html/2607.04609)|독립 noise 청크 불연속,이전 행동으로 steering|연속성 보정의 단순 재제안 방지|
|[EagleVLA](https://arxiv.org/html/2607.12659)|적응 호출·상주버퍼·graph·unroll|이후에도 남는 병목|
|[Leap-VLA](https://arxiv.org/html/2611.01011)|캐시 나이 조건화|오래된 특징 고려 자체의 중복|
|[FlashVLA](https://z-lab.ai/projects/flashvla/)|streaming denoise와 행동 연속성|이미 최적화된 실행에서 추가 비용 확인|
|[Cosmos Policy](https://arxiv.org/html/2601.16163)|행동·미래·가치 생성과 검색|선택 품질을 보존하는 분기 실행 최적화|

새 하드웨어 주제의 채택 조건:추가 계산이 실제 품질을 개선하고,강한 소프트웨어 기준선 뒤에도 비용이 남고,제안 구조의 전체 지연/트래픽/면적/전력 및 정확도를 비교할 수 있어야 합니다. 논문에서 품질 이득을 보고했다는 사실만으로 우리 구현의 이득을 확정하지 않습니다.

## 11. 파일·재현·실험 감사

전체 결과 목록은 [reports/README](../reports/README.md),상세 이전 인계는 [RESEARCH_HANDOFF](RESEARCH_HANDOFF.md),원래 결과 로그는 [RESULTS](RESULTS.md)입니다. 이 문서가 통합 요약의 기준입니다. 각 시점의 원자료는 해석 수정 후에도 지우지 않습니다.

모델·데이터·가상환경은 checkpoints/,data/,.runtime/ 및 .venv/에 있고 Git에서 제외합니다. 전체 로컬 raw는 results/에 있으며 push로 자동 업로드되지 않습니다. reports/에 수록된 JSON/CSV/코드 snapshot과 이 문서의 수치는 Git에 보존합니다. HTML의 일부 ../../results 링크는 서버에서만 열릴 수 있습니다. 영상은 simulator 시간으로 재생하며 추론 대기는 대개 생략됩니다.

보존할 오류:조기 goal-stop,명령 dtype 변환,Isaac 단위/joint 차이,자기 simulator GPU 경합,초기 compiler 호출이 섞인 trace. 이러한 실행은 유효 성공률·steady timing에서 제외하거나 오염을 명시합니다. 요청한 가설을 지지하도록 결과를 골라 합치지 않습니다.

## 12. 2026-11-19 추가 탐색

이번 요청에 따라 최신 ML의 새로운 가속 응용과 VLA 후속을 진행합니다. 완료한 실험만 아래에 추가하며,후보 검토·설치·실측·품질 검증 단계를 구분합니다.
