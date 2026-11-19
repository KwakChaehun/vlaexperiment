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
- FlashVLA는 공개 모델로 실행 가능했습니다. 후속 동일 48조건에서 padding 제거 전후 성공은 42/48→45/48, 호출 중앙값은 62.28→51.30ms였습니다. 정확도 향상을 확정하기에는 표본이 작습니다(McNemar p=.25).
- 새 응용 PISG에서는 공통 앞부분을 재사용해 전체 solver 시간이 약 21~23% 줄었고, native/shared 320쌍의 최종 경로가 같았습니다. 일반 후보를 두 개 생성하는 방식과의 품질·시간 절충도 남아 있습니다.
- TabICLv2에서는 기존 KV cache와 작은 질의의 실행 관리 최적화가 큰 효과를 냈습니다. 단일 질의를 위한 전용 하드웨어보다 여러 데이터 문맥을 유지하는 용량·스케줄 문제를 먼저 검증할 가치가 있습니다.

이 문서는 과거 실험과 이번 추가 실험을 모두 포함합니다. **새 결과만 읽으려면 12절**, 기존 GR00T 여러 후보 가속은 7절, 초기 H/K 비교는 3절을 보시면 됩니다. 서로 다른 모델·데이터·판정의 성공률을 직접 비교하지 않습니다.

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

### 작업별 및 접시 이동 세부 결과

|접촉 작업(각12조건)|H50/K16|H50/K4|H16/K4|
|---|---:|---:|---:|
|아래 서랍에 그릇 넣고 닫기|8/12|8/12|10/12|
|위 서랍을 열고 그릇 넣기|11/12|10/12|7/12|
|접시를 스토브 앞쪽으로 밀기|11/12|9/12|9/12|
|크림치즈를 그릇에 놓기|11/12|12/12|12/12|
|와인병을 랙에 놓기|12/12|11/12|9/12|

아래 서랍·크림치즈에서 H16/K4의 표본상 우위가 있었지만,같은 구조의 삽입 공통 특성으로 확정하지 못했습니다. 위 서랍은 반대 경향입니다. 과거 설명의 작업명만 보고 정밀 삽입 benchmark로 해석하면 안 됩니다.

별도의 bowl-on-plate 스트레스는 접시를 world X축으로0/±20mm 순간 이동시킨 합성 외란입니다. 각 설정6초기 상태이고 모든54회가 한 번은 목표를 통과했습니다. 유지 판정은 아래와 같아 초기 goal 통과만 보면 천장 효과가 큽니다.

|접시 이동|H50/K16 유지|H50/K4 유지|H16/K4 유지|
|---|---:|---:|---:|
|0mm|2/6|0/6|0/6|
|+20mm|1/6|0/6|0/6|
|-20mm|3/6|2/6|2/6|

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

[GR00T/π0.5/OpenVLA-OFT](../reports/large_vla_candidates_20261118/RESULTS.md):31상태에서 총186branch,첫8행동만 개입 후 동일 정책으로 이어감. 유지 성공 GR16/31,π19/31,OFT16/31,합의·그룹CV 검증18/31,정답을 아는 oracle21/31. Oracle은 배포 가능한 방식이 아닙니다.

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

### 12.1 최신 응용 후보 조사

조회 기준:2026-11-19. 학회 확인은 공식 proceedings 또는 저자 공식 저장소에 근거하며,preprint는 별도 표시합니다. 아래 하드웨어 가능성은 연구 가설이지 신규성·성능이 확정된 결론이 아닙니다.

|응용/논문|추가 계산의 역할|가속 대상으로 볼 비용|중복·위험과 현재 단계|
|---|---|---|---|
|[TabICLv2,ICML2026](https://arxiv.org/abs/2602.11139)|표 데이터를 문맥으로 넣어 분류·회귀,변환 ensemble|긴 문맥의 KV 저장/읽기,행·열 attention,짧은 반복 질의|공식 캐시·CPU/disk offload가 이미 있음. 공개 모델3090 실측 및 문맥 확장 측정 완료|
|[PISG,저자 저장소 NeurIPS2026 표기](https://arxiv.org/abs/2611.19768)|정상/구조 약화 분기를 결합해 TSP/MIS 제약에 맞는 점수 생성|두 분기의 공통 앞부분,edge feature N² 데이터,분기 이후 메모리 수명|논문 게재 상태는 저자 표기;OpenReview 직접 조회는 브라우저 challenge. 학습된 solver의 실제 가속·경로 평가 완료|
|[NExCO,ICLR2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3d681cc4487b97c08e5aa67224dd74f2-Abstract-Conference.html)|제약을 만족하는 부분해를 점진적으로 확장|불규칙 active set,feasibility projection,GNN/decoder 전환|고정 step보다 instance-adaptive scheduling이 필요할 가능성. 이번에는 논문·공개 코드 위치 확인만|
|[TTS-Design,IJCAI2026](https://ijcai.org/proceedings/2026/616)|여러 단백질 서열을 구조-서열 reward로 선택|고정 구조에서 후보별 중복 표현,후보 scorer batch|공유 가능한 특징의 의존성 검증 필요. 서열 recovery와 folding reliability 모두 확인해야 함. 이번에는 논문 조사|
|[Multi-Block Diffusion LM,2026 preprint](https://arxiv.org/abs/2606.29215)|다른 noise 단계의 여러 block을 동시에 생성|block별 종료 시점,KV commit/유지,동적 요청의 정적 buffer packing|공식 Diffulex가 이미 static buffer·prefix cache·CUDA graph 제공. LLaDA2-mini 계열은16B,SDAR계열은8B 공개. 이번에는 모델 목록/실행 경로 조사|
|[SAM3/3.1,ICLR2026 및 공식 후속](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md)|여러 개체를 비디오에서 검출·추적|개체별 메모리와 동적 개체 수 처리|3.1에서 object multiplexing을 이미 구현. 단순 backbone 공유·batch만으로 차별성 부족. 이번에는 공식 변경 조사|
|[A²WM,2026-11 preprint](https://arxiv.org/html/2611.15011)|정책 주변 action correction의 반응을 affine하게 예측하고 QP로 보정|world model1회+작은 QP,제어 마감|이미 저비용 보정이 목표인 소프트웨어 대안. 비싼 검색 가속의 필요성을 약화시킬 수 있음. 이번에는 논문 조사|
|[Cosmos Policy,ICLR2026](https://arxiv.org/html/2601.16163)|행동 후보→미래→가치 검색|분기별 다른 반복수,두 모델 가중치 상주·배칭|논문 품질 이득이 우리 환경에서 재현되는지 미확인. 이전 공식 코드 감사 유지|

Tabular 분야에도 직접 선행연구가 있습니다. [Benchmarking Attention for Tabular Foundation Models](https://arxiv.org/abs/2609.31306)는 A100/H100/B200에서 SDPA/FA2·3·4 등 backend와 행·열 attention layout 비용을 비교합니다. 따라서 "표 attention은 일반 LLM과 다르다"나 단순 backend 교체를 새 기여로 삼을 수 없습니다. [Tabular Foundation Models Are Effectively Shallow](https://arxiv.org/abs/2610.13253)는 층을 줄이는 강한 소프트웨어 대안입니다. 새로운 cache/attention 하드웨어는 이런 기준선도 고려해야 합니다.

### 12.2 TabICLv2 실제 초기 결과

공개 `jingang/TabICL/tabicl-classifier-v2-20260212.ckpt`,official source를 별도 폴더에 설치했습니다. 실제 데이터는 sklearn digits(학습1024/시험773),breast_cancer(400/169)의 stratified 고정 split이며,의료적 판단을 위한 모델 평가가 아니라 공개 분류 벤치마크입니다. Synthetic4096은 성능 scaling용이고 응용 품질 근거가 아닙니다. Ensemble1/4/8,공식 cache 없음/repr/kv를 비교하여 실제 데이터18설정+합성3설정=21설정,설정별 query1/16/128×5회=315개 warmed timing입니다. Fit(모델 load·cache 생성) 비용은 별도 기록했습니다.

|데이터,ensemble|cache 없음,1행 질의|repr cache|KV cache|전체 시험 정확도|
|---|---:|---:|---:|---:|
|digits,E1|34.11ms|9.87ms|8.90ms|99.35%|
|digits,E4|115.10ms|24.32ms|24.89ms|99.35%|
|digits,E8|202.47ms|33.26ms|37.21ms|99.48%|
|breast_cancer,E1|20.10ms|8.82ms|6.07ms|97.04%|
|breast_cancer,E4|68.69ms|21.49ms|17.49ms|97.04%|
|breast_cancer,E8|123.85ms|30.81ms|27.80ms|97.04%|
|synthetic4096,E4|367.46ms|70.15ms|25.66ms|96.88%|

14개 cache 대조에서 예측 클래스 변화0이었지만 확률은 비트 동일하지 않고 최대 차이0.00309였습니다. "정확한 cache"나 "정확도 일반 보존"을 주장하지 않습니다. Digits E8에서 repr 논리 cache119MB,KV304MB;합성4096 E4에서87MB 대456MB입니다. Logical tensor bytes는 실제 DRAM traffic이 아닙니다. 작은 문맥에서는 repr이 KV보다 빠른 경우도 있어 cache를 크게 만드는 것이 항상 유리하지 않습니다. 대부분의 큰 속도 이득은 기존 공식 소프트웨어 기능이며 새 하드웨어 결과가 아닙니다.

Raw:`results/ml_workloads_20261119/tabicl/`의 protocol.json,runs.jsonl,cache_parity.json 및 각 확률 NPZ. 코드:`scripts/probe_tabicl_workload.py`. 단일 split·작은 실제 데이터만으로 모델 간 일반적인 정확도 순위를 정하지 않습니다.

### 12.3 FlashVLA 후속: 언어 padding

동일한8개 실제 관측과 고정 action buffer를 사용하여 공식 fusion+compile 상태에서 언어 right-padding을 줄였습니다. 잘라낸 모든 토큰의 attention mask가 false임을 assertion으로 검사했습니다. 각 길이64회,총256회. 학습된 action buffer4×10과 유효 관측·언어는 유지했습니다. 시간은 준비된 GPU 입력에서 영상 처리·VLM·expert·CPU 출력 복사까지이며, CPU tokenizer/normalizer와 simulator, 최초 compile은 제외합니다.

|언어 길이 상한|전체 호출 p50|p99|200 대비 출력 비트 일치|
|---|---:|---:|---:|
|200|62.275ms|63.377ms|8/8 대조군|
|128|51.948ms|53.510ms|0/8|
|64|51.297ms|52.820ms|0/8|
|56|51.282ms|53.090ms|0/8|

유효하지 않은 토큰을 줄여도 행렬 크기·compile 경로가 바뀌면서 BF16 결과가 바뀝니다.64에서 normalized action RMSE 최대0.00380,56에서0.00463이었습니다. 이는 물리 단위 m 오차가 아닙니다.64이하에서는 추가 속도 이득이 거의 없어 closed-loop 비교는200 대64로 진행했습니다. 이 최적화는 padding 제거라는 소프트웨어 기준선이며 새 하드웨어 기여가 아닙니다.

### 12.4 TabICL 장문맥 scaling과 남은 병목

합성32feature 데이터의 문맥1024/4096/16384/32768행,ensemble4,repr/KV 두 방식,질의1/16/128을 비교했습니다.8개 설정×15개 warmed timing=120회이며,응용 정확도·일반화 성능 근거가 아닌 부하 실험입니다.

|문맥 행 수|repr 단일 질의|KV 단일 질의|fit 직후 GPU allocated repr / KV|
|---|---:|---:|---:|
|1024|23.85ms|24.37ms|0.161 / 0.254GB|
|4096|70.81ms|24.88ms|0.186 / 0.556GB|
|16384|544.55ms|29.18ms|0.286 / 1.763GB|
|32768|1911.02ms|29.38ms|0.287 / 3.374GB|

Allocated는 cache만이 아니라 모델 등의 live tensor도 포함합니다. Peak는 더 큽니다(32768 KV 약9.66GB). 공식 자동 offload 정책을 유지했으므로 작은 repr resident 값을 전체 논리 문맥 크기로 해석하면 안 됩니다. 전체 KV를 저장하면 반복 질의는 크게 빨라지지만 dataset별 상주량이 증가합니다.

16384행 단일 질의 Torch profiler:repr GPU self-time 합 약526ms 중 flash-attention 약410ms;KV 경로 GPU self-time 합 약8.51ms 중 flash-attention 약3.46ms,copy 약2.00ms였습니다. KV 쪽 전체 호출은 약29ms여서 GPU 연산 외 비용도 큽니다. CPU/GPU profiler 합은 겹칠 수 있고 profiler 오버헤드도 있으므로 단순 차감으로 하드웨어 최대 speedup을 계산하지 않습니다. 반복적인 작은 연산의 graph/제출 최적화 시도와 실패는 12.8절에 기록했습니다.

현재 판단은 **여러 dataset의 문맥을 유지하면서 드문·작은 질의를 처리하는 cache 용량/배치/상주 정책**이 조사 대상이라는 것입니다. 단일 context의 기존 KV cache 기능을 가속기 기여로 삼지는 않습니다. 기존 CPU 트리 모델 대비 필요한 정확도 이득과 dataset 교체 비용도 확인해야 합니다.

### 12.5 FlashVLA 96회 실제 실행 확인

앞선16초기 상태 비교 후 나머지32개를 추가했습니다.4작업×12초기 상태=48개 같은 장면에서 cap200과64를 각각 실행했습니다. 첫16은 탐색,후속32는 따로 저장했습니다. 단일 seed이며 학습 task와 겹칠 수 있는 공개 reset입니다. 첫 입력은 저장 원본으로 같고 이후에는 각각 실행한 실제 simulator 관측을 사용합니다.

|작업|cap200|cap64|
|---|---:|---:|
|크림치즈→그릇|10/12|10/12|
|위서랍 열고 그릇 넣기|11/12|12/12|
|아래서랍에 그릇 넣고 닫기|9/12|11/12|
|그릇→접시|12/12|12/12|
|합계|42/48|45/48|

cap64의 rescue3,harm0,관측 순증6.25%p입니다. Exact McNemar p=.25로 정확도 향상을 확정하지 못합니다. Task별 paired bootstrap95%[0,+12.5]%p는 작은 표본·불일치3건에서 얻은 구간이므로 p값과 함께 봐야 합니다. 공통 physical step925개에서 새 noise tail hash925/925가 일치했습니다. Native goal 판정이며 별도 release/hold 강화 또는 정밀삽입 지표가 아닙니다. 추론 중 물리가 멈추므로 속도 단축이 성공 증가를 유발했다는 인과 해석을 할 수 없습니다. Padding 변경의 수치 차이가 궤적에 영향을 줍니다.

**판단:**공식 최적화 이후에도 padding 제거로약17.6%의 steady 속도 이득이 남았습니다. 이 표본에서 성공률 저하는 관측하지 않았지만,전용 하드웨어나 일반 정확도 보존의 증거는 아닙니다. Raw:`results/vla_extension_20261119/flash_trim/` 및 `results/flashvla_probe_20261119/{extension,confirmation}_cap{200,64}/`.

### 12.6 PISG: 학습된 GNN의 공통 분기 재사용

공식 PISG의 StruDiCO dense 경로와 저자 공개 TSP50/100 체크포인트를 strict load했습니다.12층 hidden256 GNN에서 공식 설정인 layer7의 gate를 adaptive하게 약화시킵니다. 입력이 같은 앞7층은 한 번 계산하고,그 뒤 정상/약화 두 분기로 확장했습니다. 서로 다른 입력의 근사 cache나 새 하드웨어가 아닙니다.

초기 실행 실패 두 가지도 보존했습니다. 선택하지 않은 MIS 모듈의 torch_sparse import를 피하도록 TSP 모듈만 로드했고,dense GNN에서는 미사용 torch_sparse import만 생략했습니다. CUDA Graph 안에서 CPU→GPU timestep frequency 상수 복사가 실패하여,원래 CPU float32 수식으로 만든 상수를 미리 GPU에 유지했습니다.4timestep embedding의 비트 일치를 확인했고,baseline/shared 모두 같은 상수 최적화를 적용했습니다.

Denoiser-only:실제8문제×4timestep×2크기×3분기위치=192비교,무작위 순서216timing. Native eager와 native graph는192/192 출력 일치. 공유 방식은 행렬 batch 크기가 달라 로그잇 비트 일치가 일반적으로 깨졌지만 edge별 argmax는192/192 같은 결과였습니다. 이 사실만으로 tour 품질 보존을 단정하지 않았습니다.

|공식 layer7 조건|native eager|native CUDA Graph|공통 앞부분+Graph|
|---|---:|---:|---:|
|TSP50 denoiser|4.730ms|4.223ms|3.145ms|
|TSP100 denoiser|15.504ms|14.998ms|10.993ms|

layer0/11도 진단했습니다. TSP100에서0은15.017→14.877ms,11은14.999→8.781ms로 공유 가능한 앞부분 길이에 따라 달랐습니다. 다른 perturb layer로 바꾸면 알고리즘도 달라지므로 layer11을 공식 품질 결과에 대입하지 않습니다.

### 12.7 PISG 실제 TSP 경로 평가 및 싼 후보2개 대조군

공식5step scheduler,layer7/adaptive/scale.1/linear guidance,공식 native stitch decoder를 사용했습니다. 의도한 no-two-opt 예산을 지키기 위해 max_iterations=0이면 명시적으로 skip했습니다. 원본 helper는0이어도 while-loop에서 한 번 갱신할 수 있었습니다. 모든 비교 방식에 같은 처리를 했고 원논문 숫자의 완전 재현이라고 주장하지 않습니다.0-based 도시 index와 닫힌 Hamiltonian tour(모든 도시 한 번 방문)를 검사했습니다.

첫16문제/크기×2seed×3방식=192회. 이후 **새64문제/크기×2크기×2seed×4방식=1024회**를 추가했습니다. 아래는 새 문제만의 결과입니다. Gap은 제공된 reference tour보다 얼마나 긴지(작을수록 좋음)이며 성공률이 아닙니다. Seed 반복은 같은 문제의 상관된 표본입니다.

|문제|방식|평균 gap|전체 solver p50|
|---|---|---:|---:|
|TSP50|일반 후보1개|0.01341%|12.60ms|
|TSP50|일반 후보2개 중 짧은 경로|0.00774%|21.99ms|
|TSP50|PISG native Graph|0.01741%|20.15ms|
|TSP50|PISG 공통 앞부분 공유|0.01741%|15.90ms|
|TSP100|일반 후보1개|0.10275%|41.03ms|
|TSP100|일반 후보2개 중 짧은 경로|0.05133%|75.91ms|
|TSP100|PISG native Graph|0.08427%|68.91ms|
|TSP100|PISG 공통 앞부분 공유|0.08427%|53.04ms|

초기+후속에서 native/shared 경로 **320/320쌍 동일**,cost 차이0이었습니다. 전체1216회 경로가 유효했습니다. 전체 solver에는 CPU decoder와 GPU→CPU 등이 포함되며,model load/graph 준비는 제외합니다. 새 문제에서는 PISG 공유가 native Graph보다TSP50 약21.1%,TSP100 약23.0% 빨랐습니다.

**해석:**앞부분 공유의 실효 속도 이득은 확인했습니다. 하지만 TSP50에서는 PISG가 단일 후보보다도 나빴고,TSP100에서는 단일 후보보다 개선되나 더 비싼 best-of2가 더 좋은 해를 냈습니다. TSP는 좌표로 실제 경로 비용을 즉시 계산할 수 있어 best-of2 선택은 미래 정답 oracle을 쓰는 VLA 평가와 다릅니다. 현재는 속도·품질의 trade-off이며 PISG의 보편적 우위나 전용 가속기 필요성이 입증된 것은 아닙니다. 더 어려운 큰 sparse graph,공식 큰 batch,다양한 solver에서 검증해야 합니다.

Raw:`results/ml_workloads_20261119/pisg/`, `pisg_solver/`, `pisg_confirmation/`. 최초 오류 로그도 별도 유지합니다. 학습 데이터의 일반화·원논문 전체 benchmark·area/power/DRAM 실측은 수행하지 않았습니다.

### 12.8 TabICL 추가 소프트웨어·CPU 대조군

원래 cached forward를 그대로 CUDA Graph에 넣는 시도는 실패했습니다. 공식 inference manager가 실행마다 GPU memory 조회 전에 synchronize를 호출하기 때문입니다. 알려진 작은 질의에 한정해 fit 이후 `auto_batch=False`를 사용하는 추가 실험을 했습니다. 기존 모델의 연산식·가중치는 유지하고 같은 입력의 확률을 비교했습니다.

|Digits,E4,KV cache|공식 자동배치 p50|알려진 크기 고정배치 p50|출력 확률|
|---|---:|---:|---|
|질의1행|24.75ms|12.34ms|최대 차이5.14e-10|
|질의16행|20.50ms|13.43ms|비트 일치|
|질의128행|22.53ms|16.18ms|비트 일치|

각 경우10회,warm-up 별도입니다. 임의로 큰 query에 적용하면 메모리 자동조절을 잃으므로 현재 검증한 크기에만 해당합니다. 고정배치 경로의 추가 graph capture도 실패했으므로 graph speedup을 보고하지 않습니다. Raw traceback을 저장했습니다. 이는 전체 하드웨어 가속보다 실행 관리 비용을 먼저 줄여야 할 수 있다는 근거입니다.

동일 stratified split의 기본 CPU 모델도 비교했습니다. 튜닝한 최강 tree 기준선은 아닙니다.

|데이터|모델|정확도|CPU 학습|1행 질의 p50|
|---|---|---:|---:|---:|
|digits|HistGradientBoosting|96.38%|847.94ms|6.87ms|
|digits|RandomForest300|97.67%|241.03ms|21.10ms|
|breast_cancer|HistGradientBoosting|94.67%|89.09ms|0.744ms|
|breast_cancer|RandomForest300|94.08%|192.07ms|21.08ms|

이번 split에서 TabICL의 정확도는 더 높았지만 한두 작은 dataset의 차이를 전체 응용 수요로 일반화하지 않습니다. RF의 단일 질의는2thread 스케줄 비용에도 영향을 받으므로 최적 CPU 지연이라고 주장하지 않습니다.

### 12.9 이번 추가 탐색의 최종 우선순위

1. **PISG/NExCO 계열의 불규칙 graph·분기 실행:**새 응용으로 가장 직접적인 후속 후보입니다. PISG 공통 prefix의 소프트웨어 이득을 먼저 제거한 뒤,큰 sparse graph의 gather/scatter·edge tensor lifetime·제약 decoder 전환 비용을 측정해야 합니다. 작은 TSP50/100의 이득만으로 하드웨어 설계를 확정하지 않습니다.
2. **Tabular foundation model의 여러 context 상주와 소량 질의:**문맥 크기에 따른 cache trade-off는 확인했습니다. 다만 공식 KV와 고정배치만으로 큰 이득이 있어,query arrival·여러 dataset 교체·메모리 용량을 포함한 실제 workload가 있어야 전용 하드웨어의 근거가 강해집니다. 일반 row/column attention backend 비교는 직접 선행연구가 있습니다.
3. **FlashVLA 및 VLA:**기존 최적화 뒤 padding 제거로약17.6% 단축,48조건에서42→45성공이었지만 정확도 증가의 통계 근거는 약합니다. Streaming 전체의 지연·관측 나이와 성공을 함께 평가할 실제시간 실험이 남았습니다.
4. **Cosmos 검색·protein reward 후보선택·MultiBD:**계산 구조가 유망하나 이번 요청에서 실제 모델을 실행하지 않았습니다. 발표 성능을 우리 실측으로 표시하지 않습니다. 학습된 scorer가 싼 후보나 작은 모델보다 추가 품질을 주는지부터 확인할 대상입니다.

이번에 새로 실행한 범위:TabICL29개 설정/435 warmed timing+고정배치60 timing+CPU대조군4개,FlashVLA padding256 timing와96개 policy rollout,PISG192개 denoiser 출력 비교/216timing 및1216개 solver 실행. 재사용한 데이터·문제와 두 seed를 독립적인 새 데이터로 합산하지 않습니다. 새로운 PIM/ASIC/FPGA 또는 area/power 평가를 수행한 것은 아닙니다.

하드웨어 주제로 발전시킬 때의 다음 검증은 아래와 같습니다. **아직 수행한 결과가 아니라, 이번 실측에서 도출한 연구 가설과 중단 기준**입니다.

|후속 가설|먼저 비교할 강한 GPU 기준선|유효한 연구 근거 / 중단 기준|
|---|---|---|
|큰 graph의 정상·약화 분기를 처리하는 과정에서 edge 데이터 이동이 병목이다|공통 앞부분 공유, CUDA Graph, sparse kernel 및 후보 batch|실제 solver의 남은 시간이 gather/scatter·데이터 이동에 집중되는지 확인. 단일 후보 또는 일반 best-of가 같은 시간에 더 좋은 해를 내면 해당 PISG 부하의 우선순위를 낮춤|
|여러 표 데이터의 context 교체가 소량 질의의 지연을 지배한다|공식 KV cache, 고정배치, GPU 상주·CPU offload 정책|서로 다른 데이터셋 요청이 섞인 상태에서 cache 전송량과 p99를 측정. 모두 GPU에 상주하며 기존 방식으로 지연 목표를 만족하면 전용 하드웨어 필요성은 약함|
|VLA streaming에서 계산 지연을 줄이면 실제 접촉 결과가 좋아진다|padding 제거와 공식 compile을 적용한 FlashVLA|같은 행동·noise 조건에서 물리를 실시간으로 진행하고 관측 나이, 마감 초과, release/hold 결과를 함께 비교. 속도만 빨라지고 접촉 결과가 유지되거나 악화되면 정확도 이득을 주장하지 않음|

## 13. 현재 저장·푸시 상태와 후속 재현

2026-11-19 최초 통합 문서는 커밋 `a6bd785`로 origin/main에서 확인했습니다. 이 후속 문서에도 **통합 Markdown 한 파일만** 커밋·푸시하는 범위를 유지합니다. 추가 결과는 12절에 합쳤습니다.

상세 새 실행 코드는 scripts/probe_tabicl_workload.py,probe_tabicl_scaling.py,probe_tabicl_fixed_batch.py,probe_flashvla_trim.py,evaluate_flashvla_extension.py,probe_pisg_prefix.py,evaluate_pisg_solver.py,evaluate_pisg_confirmation.py,analyze_new_workloads.py 등에 있습니다. 현재 요청에서 모델·데이터·원자료·추가 스크립트를 모두 GitHub에 올린 것은 아닙니다. 이 파일은 GitHub에서도 수치·판정·한계를 읽을 수 있도록 자체 완결형으로 작성했습니다. 아직 Git에 없는 상세 보고서 링크는 로컬 workspace에서만 열립니다.

최종 감사에서 TSP 1,216회 경로의 도시 범위·중복·닫힌 경로 여부를 다시 검사하고, 좌표에서 비용과 gap을 독립 재계산했습니다. 저장된 비용과 최대 차이 5.33e-15, gap 차이 1.11e-13%p였습니다. 새로운 Python 소스 13개는 구문 검사를 통과했습니다. 집계는 `results/ml_workloads_20261119/final_summary.json`, 감사·최종 소스 SHA256는 `final_audit.json`에 보존했습니다. 실행 중 수정한 loader와 이후 metadata 수정은 로그에 남겨, 최종 파일이 모든 과거 실행과 byte 단위로 같다고 주장하지 않습니다.

|공식 소스|검사한 revision|
|---|---|
|TabICL|`7ccaddaa45f9d586d90df37126d39a0cb2a964e7`|
|PISG|`19b1dd22fe219e6855211149bb0b30e18ee25c61`|
|StruDiCO|`538783956830e8532c77bea316f86aacbb6058a4`|
|FlashVLA|`8539f1cdfcf878e3a93c39f3d6c885c19ede6af8`|
