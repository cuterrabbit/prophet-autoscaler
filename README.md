# Prophet Autoscaler

BNPL 농자재 플랫폼의 농업 계절성을 반영한 **시계열 예측 기반 Pod 자동 스케일링** PoC

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [아키텍처](#2-아키텍처)
3. [디렉토리 구조](#3-디렉토리-구조)
4. [더미 데이터 설계](#4-더미-데이터-설계)
5. [모델 설계](#5-모델-설계)
6. [Pod 수 계산 로직](#6-pod-수-계산-로직)
7. [모델 최적화: 로그 변환](#7-모델-최적화-로그-변환)
8. [하이퍼파라미터 튜닝 (그리드 서치)](#8-하이퍼파라미터-튜닝-그리드-서치)
9. [Optuna 최적화 + yhat_upper Pod 전략](#9-optuna-최적화--yhat_upper-pod-전략)
10. [검증 방법](#10-검증-방법)
11. [최종 성능 비교](#11-최종-성능-비교)
12. [실행 방법](#12-실행-방법)
13. [연간 예측과 KEDA 연동](#13-연간-예측과-keda-연동)

---

## 1. 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 목적 | 농업 계절성(파종/수확/기상)을 반영한 트래픽 예측으로 Pod 선제 스케일링 |
| 방식 | Reactive(HPA) → Predictive(Prophet + KEDA) |
| 예측 대상 | 요청수(건/분), CPU 사용률(%) |
| 스케일링 범위 | MIN 1 Pod ~ MAX 8 Pod |

### Reactive vs Predictive

```
[Reactive - HPA]
트래픽 급증 → CPU 상승 → HPA 감지 → Pod 증가 (이미 늦음)

[Predictive - Prophet + KEDA]
Prophet이 트래픽 예측 → Prometheus Custom Metric Push
→ KEDA가 감지 → Pod 선제 증가 (트래픽 오기 전 준비 완료)
```

---

## 2. 아키텍처

```
[연 1회]
Prophet 학습 → 연간 예측(8,760시간) → PostgreSQL 저장

[매시간 CronJob]
PostgreSQL 조회 (현재 시각의 required_pods)
        ↓
Prometheus Custom Metric Push
prophet_required_pods{service="request-rate"} = 5
        ↓
KEDA ScaledObject 감지
        ↓
Deployment replicas = 5
```

### KEDA ScaledObject 예시

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: prophet-autoscaler
spec:
  scaleTargetRef:
    name: api-deployment
  minReplicaCount: 1
  maxReplicaCount: 8
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus:9090
        metricName: prophet_required_pods
        query: prophet_required_pods{service="request-rate"}
        threshold: "1"
```

---

## 3. 디렉토리 구조

```
prophet-autoscaler/
├── data/                              # 생성된 데이터 / 예측 결과 / 그래프
│   ├── dummy_request_rate.csv         # 학습 데이터 (2년치, 17,521행)
│   ├── dummy_cpu_utilization.csv      # CPU 학습 데이터
│   ├── dummy_anomaly_events.csv       # 이상치 시나리오 정보
│   ├── predictions_request_rate.csv   # 24시간 예측 (baseline)
│   ├── predictions_annual_2025.csv    # 연간 예측 (baseline)
│   ├── predictions_log_*.csv          # 예측 결과 (최적화)
│   └── plot_*.png                     # 시각화 결과
│
├── models/                            # 학습된 모델
│   ├── request-rate-forecast.pkl      # baseline 모델
│   ├── request-rate-forecast-log.pkl  # 최적화 모델 (로그 변환)
│   └── runs.json                      # 실험 기록 (MLflow 대체)
│
├── src/                               # baseline 구현
│   ├── generate_dummy_data.py         # 모듈 1: 더미 데이터 생성
│   ├── train_predict.py               # 모듈 2: 학습 / 예측
│   ├── retrain.py                     # 모듈 3: 재학습 파이프라인
│   ├── compare_plot.py                # 시각화
│   └── validate.py                    # 검증 (시각 검증 + 홀드아웃)
│
├── src_optimized/                     # 최적화 구현 (로그 변환)
│   ├── train_predict.py               # 로그 변환 학습 / 예측
│   ├── validate.py                    # baseline 대비 성능 비교
│   └── compare_plot.py                # 시각화 (baseline vs 최적화 비교)
│
├── src_tuned/                         # 튜닝 구현 (로그 변환 + HP 최적화)
│   ├── hyperparameter_search.py       # 그리드 서치 (조합 단위 병렬화)
│   ├── train_predict.py               # 탐색 결과 자동 로드 후 학습
│   ├── validate.py                    # baseline / log / tuned 3-way 비교
│   └── compare_plot.py                # 시각화 (3-way 비교 + HP 히트맵)
│
├── src_optuna/                        # Optuna 구현 (TPE 베이지안 + yhat_upper Pod)
│   ├── hyperparameter_search.py       # Optuna TPE 탐색 (멀티프로세스 병렬)
│   ├── train_predict.py               # 탐색 결과 로드 + yhat_upper 기반 Pod
│   ├── validate.py                    # baseline / log / tuned / optuna 4-way 비교
│   └── compare_plot.py                # 시각화 (4-way 비교 + trial 산점도)
│
└── module-spec/                       # 모듈 명세서
    ├── module-spec-1-2.md             # 모듈 1, 2 명세
    └── module-spec-3.md               # 모듈 3 명세
```

---

## 4. 더미 데이터 설계

실제 Prometheus 데이터 없이 농업 서비스 특화 패턴으로 생성 (2023-01-01 ~ 2024-12-31)

### 계절 패턴 (Yearly)

BNPL 농자재 플랫폼 특성상 파종 전 농자재 구매가 피크, 수확 후 BNPL 상환 조회 트래픽 발생

```
1월  ▂  (0.20) 휴경
2월  ▃  (0.30) 다음 시즌 준비
3월  ██ (1.00) 고추·사과·감귤 파종 — 연중 최고점
4월  ▇  (0.85) 쌀·고추 파종 마무리
5월  ▅  (0.60) 모내기 + 양파 상환
6월  ▄  (0.50) 양파 상환 + 장마 진입
7월  ▃  (0.35) 장마철 불규칙
8월  ▄  (0.45) 고추 상환 + 병충해 방제
9월  ▄  (0.45) 쌀·고추 상환 + 양파 파종
10월 ▃  (0.35) 쌀·사과 상환
11월 ▃  (0.35) 사과·감귤 상환
12월 ▂  (0.20) 감귤 상환, 휴경
```

### 요일 패턴 (Weekly)

| 요일 | 가중치 | 이유 |
|------|--------|------|
| 월~금 | 1.00 | 농부 활동 집중 |
| 토 | 0.70 | 주말 거래 |
| 일 | 0.40 | 휴식 |

### 시간대 패턴 (Daily)

| 시간대 | 가중치 | 이유 |
|--------|--------|------|
| 06~09시 | 0.90 | 농작업 시작 전 주문 확인 |
| 09~12시 | 0.60 | 오전 구매 |
| 12~14시 | 0.30 | 점심 |
| 14~18시 | 0.55 | 오후 구매 |
| 18~21시 | 0.85 | 저녁 구매/상환 조회 피크 |
| 21~06시 | 0.10 | 야간 |

### 기상 Regressor

| 항목 | 기간 | 컬럼 | 영향 |
|------|------|------|------|
| 장마 | 6/25 ~ 7/25 | `is_monsoon` (0/1) | ±20% 불규칙 |
| 태풍 | 8~9월 | `typhoon_index` (0.0~1.0) | 당일 급감 → 이후 급증 |

### 이상치 시나리오 (RCA 테스트용)

| 시나리오 | 구간 | 효과 |
|----------|------|------|
| 1. 파종기 급증 | 매년 3월 10~20일 | +40% |
| 2. 태풍 직후 급증 | 매년 9월 2~4일 | +50% |
| 3. 장마 불규칙 | 첫해 7월 5~15일 | ×0.5~1.5 랜덤 |

---

## 5. 모델 설계

```python
Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True,
    seasonality_mode="additive",
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0,
)
.add_country_holidays(country_name="KR")
.add_regressor("is_monsoon")
.add_regressor("typhoon_index")
```

---

## 6. Pod 수 계산 로직

### 설계 기준

| 항목 | 값 | 근거 |
|------|----|------|
| MIN_PODS | 1 | 야간·비수기 최소 가용 |
| MAX_PODS | 8 | 3월 파종기 피크 기준 |
| CAPACITY_PER_POD | 21.1 건/분 | 피크 RPS(135) ÷ MAX_PODS(8) ÷ (1 - 0.2) |
| SAFETY_MARGIN | 0.2 (20%) | 안전 여유분 |

```
CAPACITY_PER_POD = peak_rps / MAX_PODS / (1 - SAFETY_MARGIN)
                 = 135 / 8 / 0.8 ≈ 21.1 건/분
```

### 변환 공식

```python
effective_capacity = CAPACITY_PER_POD * (1 - SAFETY_MARGIN)  # 16.9 건/분
required_pods = ceil(predicted_rps / effective_capacity)
required_pods = max(MIN_PODS, min(required_pods, MAX_PODS))   # 1~8 clamp
```

---

## 7. 모델 최적화: 로그 변환

### 문제

원본 데이터의 RPS 범위가 **0.8 ~ 135 건/분 (169배 차이)**

Prophet additive 모드는 계절 성분을 고정 절댓값으로 더하는 방식이라,
이렇게 극단적인 범위에서는 평균(~17 건/분)에 수렴해버려 계절성을 제대로 학습하지 못함

```
원본 예측 문제:
  trend ≈ 17 건/분 (전체 평균)
  3월 피크 예측: 17 + 계절성 ≈ 15 건/분  ← 실제 135와 큰 차이
  yhat_lower 음수 발생 (-28 건/분)
```

### 해결: log1p 변환

```python
# 학습 전
df["y"] = np.log1p(df["y"])   # 0.8 → 0.59, 135 → 4.91

# 예측 후
forecast["yhat"] = np.expm1(forecast["yhat"])  # 역변환
```

**변환 효과:**

| | 원본 | 로그 변환 |
|--|------|---------|
| 최솟값 | 0.8 건/분 | 0.59 |
| 최댓값 | 135 건/분 | 4.91 |
| 범위 비율 | **169배** | **8배** |
| Prophet 학습 | 평균에 수렴 | 계절성 정상 학습 |
| 음수 예측 | 발생 | 없음 |

**왜 효과적인가:**

```
log는 큰 값은 작게, 작은 값은 상대적으로 크게 압축

원본: 0.8 → 135  (간격이 너무 불균등)
로그: 0.59 → 4.91 (균등하게 압축)

→ Prophet이 3월(4.91)과 12월(0.59)의 차이를 정확히 학습
→ 예측 후 expm1으로 역변환하면 원래 단위 복원
```

### 성능 비교 (baseline → 로그 변환)

| 지표 | baseline | 로그 변환 | 개선 |
|------|---------|---------|------|
| SMAPE (교차검증) | 0.6401 | **0.2800** | 56% 개선 |
| 연간 Pod 범위 | 1~2개 | **1~8개** | 계절성 정상 반영 |
| SMAPE (시각 검증) | 0.6957 | 0.5052 | 27% 개선 |
| Pod 정확도 (시각 검증) | 68.2% | **74.6%** | +6.4% |
| SMAPE (홀드아웃) | 0.7566 | 0.6030 | 20% 개선 |
| Pod 정확도 (홀드아웃) | 69.8% | **71.2%** | +1.4% |
| 과소 프로비저닝 (홀드아웃) | 17.2% | 27.0% | ⚠️ 증가 |

---

## 8. 하이퍼파라미터 튜닝 (그리드 서치)

### 배경

로그 변환 후에도 홀드아웃 과소 프로비저닝이 27%로 높게 유지됨.
모델 파라미터 자체를 교차검증으로 탐색해 추가 개선 시도.

### 탐색 대상 파라미터

| 파라미터 | 역할 | 탐색 범위 |
|----------|------|----------|
| `changepoint_prior_scale` | 트렌드 유연성. 클수록 피크 추종, 작을수록 안정적 | 0.01, 0.05, 0.1, 0.2, 0.3 |
| `seasonality_prior_scale` | 계절성 강도. 클수록 계절 패턴 강조 | 5.0, 10.0, 15.0, 20.0 |

총 **20개 조합**을 각각 교차검증으로 평가.

### 교차검증 설정

```python
cross_validation(
    model,
    initial="365 days",   # 최초 학습 기간 (1년)
    period="30 days",     # cutpoint 간격 (30일마다 새 평가)
    horizon="7 days",     # 각 cutpoint에서 예측 기간 (7일)
    disable_tqdm=True,
)
```

```
cutpoint 생성 예시 (2년치 데이터 기준):
  학습: 2023-01-01 ~ 2023-12-31 → 예측: 2024-01-01~07
  학습: 2023-01-01 ~ 2024-01-30 → 예측: 2024-01-31~02-06
  ...
  총 약 12개 cutpoint → 각각 독립 학습 및 평가
```

평가 지표: **SMAPE** (Symmetric Mean Absolute Percentage Error)

### 병렬화 전략

20개 조합 × 12 cutpoint = 240회 학습은 순차 실행 시 약 40분 소요.
`ProcessPoolExecutor`로 조합 단위 병렬화 → **약 3배 단축**.

```python
# cutpoint 내부 병렬(parallel="processes")보다 조합 단위 병렬이 유리한 이유:
# Windows의 spawn 방식은 프로세스 생성 비용이 크기 때문에
# 풀을 20번 생성/소멸하는 것보다 1번만 생성하는 편이 효율적
N_WORKERS = os.cpu_count() - 1

with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
    futures = {pool.submit(evaluate_combo, args) for args in grid_args}
    for fut in as_completed(futures):
        result = fut.result()
```

### 탐색 결과

```
  cps     sps     SMAPE
  0.050  15.0    0.XXXX  ★ 최적
  0.050  20.0    0.XXXX
  0.100  15.0    0.XXXX
  ...
```

최적 파라미터: **`changepoint_prior_scale=0.05`, `seasonality_prior_scale=15.0`**

> 탐색 결과는 `data/hp_search_results.csv`에 저장되며,
> `src_tuned/train_predict.py` 실행 시 자동으로 로드됩니다.

### 파라미터 해석

- **cps=0.05**: 기본값(0.05)과 동일 — 트렌드를 안정적으로 유지하는 것이 유리
- **sps=15.0**: 기본값(10.0)보다 높음 — 농업 계절성이 강하므로 계절 패턴에 더 많은 유연성 부여

---

## 9. Optuna 최적화 + yhat_upper Pod 전략

### Optuna TPE (베이지안 최적화)

그리드 서치(20개 이산 조합)의 한계를 극복하기 위해 Optuna TPE(Tree-structured Parzen Estimator)를 도입.

| 항목 | 그리드 서치 (src_tuned) | Optuna TPE (src_optuna) |
|------|------------------------|-------------------------|
| 탐색 방식 | 전수 탐색 (20조합 고정) | 유망 구간 집중 탐색 (40 trial) |
| 탐색 공간 | 이산 (5×4 격자) | 연속 (log-uniform, uniform) |
| 탐색 파라미터 수 | 2개 | **3개** (holidays_prior_scale 추가) |
| 병렬화 | ProcessPoolExecutor (조합 단위) | ask()/tell() + ProcessPoolExecutor |

#### 탐색 파라미터

| 파라미터 | 역할 | 탐색 공간 |
|----------|------|----------|
| `changepoint_prior_scale` | 트렌드 유연성 | log-uniform [0.001, 0.5] |
| `seasonality_prior_scale` | 계절성 강도 | uniform [1.0, 30.0] |
| `holidays_prior_scale` | 공휴일 가중치 (신규) | uniform [1.0, 20.0] |

#### 멀티프로세스 병렬화 전략

`study.optimize(n_jobs=...)` 는 내부적으로 threading을 사용하는데, Prophet은 Stan 기반 MCMC이므로 진짜 병렬 효과를 얻으려면 별도 프로세스가 필요함.

```python
# ask()/tell() 패턴으로 ProcessPoolExecutor와 연결
with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
    while completed < N_TRIALS:
        # 배치만큼 파라미터 제안 (결과 보고 전)
        trials = [study.ask() for _ in range(BATCH_SIZE)]

        # 배치를 프로세스로 병렬 실행
        futures = {pool.submit(evaluate_params, args): trial
                   for trial, args in zip(trials, batch_args)}

        for fut in as_completed(futures):
            smape = fut.result()
            study.tell(futures[fut], smape)  # TPE 갱신
```

`constant_liar=True` 옵션: 배치 내 미완료 trial을 현재 best로 가정 → 같은 구간에 중복 탐색 방지

#### 최적 파라미터

```
changepoint_prior_scale = 0.0399   (log-uniform 공간에서 찾은 값)
seasonality_prior_scale = 9.1384
holidays_prior_scale    = 11.3258  (그리드 서치에는 없던 파라미터)
SMAPE (교차검증)         = 0.XXXX
```

---

### yhat_upper 기반 Pod 계산

#### 기존 방식의 문제 (yhat 기반)

Prophet은 점 추정치(`yhat`)와 함께 80% 신뢰 구간(`yhat_lower`, `yhat_upper`)을 출력함.
기존 코드는 `yhat` 기준으로 Pod을 결정하므로, 실제 트래픽이 `yhat`을 초과하면 Pod 부족 → 장애.

```
실제 트래픽 ──────────────────────── 55 req/min
                                      ↑ 장애 발생
yhat_upper ──────────────────── 52.8  (신뢰구간 상한)
yhat       ──────────── 45.2          (점 추정치 — 기존 기준)
```

#### 변경 (yhat_upper 기반)

```python
# 기존
forecast["required_pods"] = forecast["yhat"].apply(_to_pods)

# 변경 — 신뢰구간 상한으로 Pod 수 결정
forecast["required_pods"] = forecast["yhat_upper"].apply(_to_pods)
```

실제 트래픽이 `yhat_upper`까지 올라와도 Pod이 이미 준비돼 있음.

#### 트레이드오프

| 방식 | 과소 프로비저닝 | 과잉 프로비저닝 | Pod 정확도 |
|------|----------------|----------------|------------|
| yhat 기반 (tuned) | 26.8% ⚠️ | 1.8% | 71.4% |
| yhat_upper 기반 (optuna) | **4.0%** ✅ | 37.9% | 58.1% |

BNPL 결제 플랫폼에서 **과소(장애) > 과잉(비용)** 이므로 과잉 증가를 감수하고 과소를 억제하는 것이 합리적.

---

## 10. 검증 방법

### 시각 검증 (Fit Check)

전체 2년치(2023~2024)로 학습 후 동일 기간 재예측 → 실제 y vs yhat 비교

- 모델이 학습한 패턴(3월 피크, 야간 최저, 장마 불규칙)을 제대로 재현하는지 확인
- 학습 데이터에 대한 성능이므로 낙관적 수치 — 실 성능 지표로는 사용하지 않음

### 홀드아웃 검증 (Holdout)

2023년 데이터만으로 학습 → 2024년 예측 → 2024년 실제값과 비교

- 모델이 **본 적 없는 데이터**에 대한 일반화 성능 측정
- 실제 운영 환경을 가장 잘 모사하는 검증 방식
- 과소 프로비저닝은 장애로 직결되므로 이 지표를 가장 중점적으로 관찰

### 교차검증 (Cross Validation)

Prophet 내장 `cross_validation()`을 이용해 여러 시점(cutpoint)에서 반복 평가

```
initial="365 days" → 최초 1년치로 학습
period="30 days"   → 30일마다 cutpoint 이동
horizon="7 days"   → 각 cutpoint에서 7일 앞 예측

결과: 약 12개 시점의 SMAPE를 평균 → 단일 홀드아웃보다 안정적인 평가
```

하이퍼파라미터 탐색 시 사용. 단, 데이터 전체 기간이 2년이어서 cutpoint가 12개로 제한적.

### 지표 정의

| 지표 | 의미 | 위험도 |
|------|------|--------|
| SMAPE | 예측 오차율 (낮을수록 좋음, 임계값 0.15) | - |
| Pod 정확도 | 예측 Pod = 실제 Pod 일치율 | - |
| 과잉 프로비저닝 | 예측 Pod > 실제 Pod (비용 낭비) | 낮음 |
| 과소 프로비저닝 | 예측 Pod < 실제 Pod (서비스 장애 위험) | **높음** |

---

## 11. 최종 성능 비교

### 4-way 비교표

| 지표 | baseline | 로그 변환 | tuned | **optuna** | 비고 |
|------|---------|---------|-------|------------|------|
| **[시각 검증]** | | | | | |
| SMAPE | 0.6957 | 0.5052 | 0.5052 | 0.5052 | 동일 (fit 특성) |
| Pod 정확도 | 68.2% | 74.6% | 74.6% | 45.8% | yhat_upper로 과잉 증가 |
| 과잉 프로비저닝 | 13.4% | 9.2% | 9.2% | **53.2%** | yhat_upper 의도적 상승 |
| 과소 프로비저닝 | 18.4% | 16.2% | 16.2% | **1.0%** ✅ | 대폭 감소 |
| **[홀드아웃]** | | | | | |
| SMAPE | 0.7566 | 0.6030 | 0.5974 | **0.5969** | 최고 성능 |
| Pod 정확도 | 69.8% | 71.2% | 71.4% | 58.1% | yhat_upper 보수적 예약 반영 |
| 과잉 프로비저닝 | 5.4% | 1.8% | 1.8% | **37.9%** | 비용 증가 트레이드오프 |
| 과소 프로비저닝 | 17.2% | 27.0% | 26.8% | **4.0%** ✅ | 핵심 목표 달성 |

### 버전별 특징 요약

| 버전 | 핵심 기법 | 목적 |
|------|-----------|------|
| baseline | Prophet 기본 | 베이스라인 측정 |
| 로그 변환 | log1p/expm1 | 169배 범위 압축 → 계절성 학습 |
| tuned | 그리드 서치 HP 최적화 | SMAPE 소폭 개선 |
| **optuna** | TPE 연속 탐색 + yhat_upper | **과소 프로비저닝 26.8% → 4.0%** |

### 과소 프로비저닝 개선 흐름

```
baseline  → 로그 변환  → tuned   → optuna (yhat_upper)
  17.2%      27.0%       26.8%        4.0%
  ※ 로그 변환 후 증가는 고값 압축 부작용 (피크 과소 예측)
  ※ optuna의 yhat_upper 전략으로 Pod 계산 레이어에서 직접 해소
```

### optuna 최적 파라미터

```
changepoint_prior_scale = 0.0399   (그리드 범위 밖 연속값)
seasonality_prior_scale = 9.1384
holidays_prior_scale    = 11.3258  (그리드 서치에 없던 파라미터)
```

### 트레이드오프 판단

```
과잉 프로비저닝 37.9%  →  Pod을 여분으로 더 띄움 (비용 증가)
과소 프로비저닝  4.0%  →  장애 위험 거의 없음

BNPL 결제 플랫폼:
  과소 시 → 파종기 결제 불가 → 매출 손실 + 농가 신뢰 저하
  과잉 시 → Pod 유지 비용 증가 (감수 가능)

결론: optuna + yhat_upper 전략이 운영 관점에서 가장 안전한 선택
```

---

## 12. 실행 방법

### 전체 파이프라인 (baseline)

```bash
# 1. 더미 데이터 생성
python src/generate_dummy_data.py

# 2. 학습 + 24시간 예측 + 연간 예측
python src/train_predict.py

# 3. 모델 검증
python src/validate.py

# 4. 시각화
python src/compare_plot.py
```

### 최적화 버전 (로그 변환)

```bash
# 1. 더미 데이터는 동일하게 사용

# 2. 최적화 학습 + 예측
python src_optimized/train_predict.py

# 3. baseline 대비 성능 비교
python src_optimized/validate.py

# 4. 시각화 (baseline vs 최적화 비교 포함)
python src_optimized/compare_plot.py
```

### 튜닝 버전 (로그 변환 + HP 최적화)

```bash
# 1. 하이퍼파라미터 탐색 (20개 조합 병렬 교차검증, 약 15~20분)
python src_tuned/hyperparameter_search.py
#    → data/hp_search_results.csv

# 2. 최적 파라미터로 학습 + 연간 예측
python src_tuned/train_predict.py
#    → models/request-rate-forecast-tuned.pkl
#    → data/predictions_tuned_annual_2025.csv

# 3. 3-way 검증 (baseline / log / tuned 비교)
python src_tuned/validate.py
#    → data/plot_tuned_validation_fit.png
#    → data/plot_tuned_validation_holdout.png

# 4. 시각화 (3-way 비교 + HP 탐색 히트맵)
python src_tuned/compare_plot.py
#    → data/plot_tuned_3way_pods.png
#    → data/plot_tuned_hp_search.png
```

> `hyperparameter_search.py` 없이 `train_predict.py`만 실행하면
> 기본값 `cps=0.1, sps=10.0`으로 동작합니다.

### Optuna 버전 (TPE 베이지안 + yhat_upper Pod)

```bash
# 1. Optuna TPE 탐색 (40 trial, 멀티프로세스 병렬, 약 15~25분)
python src_optuna/hyperparameter_search.py
#    → data/optuna_search_results.csv
#    → data/optuna_study.pkl

# 2. 최적 파라미터로 학습 + yhat_upper 기반 연간 예측
python src_optuna/train_predict.py
#    → models/request-rate-forecast-optuna.pkl
#    → data/predictions_optuna_annual_2025.csv

# 3. 4-way 검증 (baseline / log / tuned / optuna 비교)
python src_optuna/validate.py
#    → data/plot_optuna_validation_fit.png
#    → data/plot_optuna_validation_holdout.png

# 4. 시각화 (4-way 비교 + Optuna trial 산점도)
python src_optuna/compare_plot.py
#    → data/plot_optuna_4way_pods.png
#    → data/plot_optuna_trials.png
```

> `optuna` 패키지 필요: `pip install optuna`

### 재학습 (추가 데이터 발생 시)

```bash
# 추가 데이터 생성 (예: 2025년 상반기)
python src/generate_dummy_data.py --start 2025-01-01 --end 2025-06-30

# 재학습
python src/retrain.py
```

---

## 13. 연간 예측과 KEDA 연동

### 왜 연간 예측인가

```
매시간 Prophet 재실행: 학습된 모델 로드 + 예측 = 1~2분 소요
연간 예측 저장:       연 1회 생성 → DB 조회(1행) = 1초 이내
```

### 저장 전략

```
[연 1회]
predictions_log_annual_2025.csv (8,760행, ~600KB)
        ↓
PostgreSQL 저장 (ds 컬럼 인덱스)

[매시간 CronJob]
SELECT required_pods FROM predictions WHERE ds = date_trunc('hour', NOW())
        ↓
Prometheus Custom Metric Push
        ↓
KEDA 스케일링
```

### 예상 월별 Pod 수

```
1월  ▂  1~2 pods  (휴경)
2월  ▃  2~3 pods
3월  ██  6~8 pods  ← 연중 최고 (파종기 피크)
4월  ▇  5~7 pods
5월  ▅  3~5 pods
6월  ▄  3~4 pods  (장마 진입)
7월  ▃  2~3 pods  (장마 불규칙)
8월  ▄  3~4 pods
9월  ▄  3~5 pods  (태풍 이상치 반영)
10월 ▃  2~3 pods
11월 ▃  2~3 pods
12월 ▂  1~2 pods  (휴경)
```
