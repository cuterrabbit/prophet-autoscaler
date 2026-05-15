# 모듈 명세서

---

## 모듈 1: 더미 데이터 생성 모듈

### 1-1. 개요

| 항목 | 내용 |
|------|------|
| 목적 | 실제 Prometheus 데이터 없이 농업 서비스 특화 시계열 더미 데이터 생성 |
| 입력 | 생성 기간, 서비스 목록, 패턴 설정값 |
| 출력 | CSV 파일 (Prophet 학습용 형식) |
| 실행 방식 | 수동 실행 (PoC 초기 1회) |

---

### 1-2. 생성할 데이터 종류

| 데이터 | 단위 | 용도 |
|--------|------|------|
| 요청수 (request_rate) | 건/분 | On-Prem Pod 스케일링 예측 |
| CPU 사용률 (cpu_utilization) | % | Cloud 노드 스케일링 예측 |

---

### 1-3. 패턴 설계

#### 작물별 계절 패턴 (Yearly)

BNPL 농자재 플랫폼 특성상 **농자재 구매 시점(파종 전)**이 트래픽 피크이고, 수확기는 **BNPL 상환 조회** 트래픽만 발생한다.

| 작물 | 농자재 구매 피크 | 상환 조회 피크 | 주요 농자재 |
|------|----------------|---------------|-------------|
| 양파 | 9~10월 (파종) | 5~6월 (수확 후) | 씨앗, 비료 |
| 쌀 | 4~5월 (모내기) | 9~10월 (수확 후) | 비료, 농약 |
| 고추 | 3~4월 (파종) | 8~9월 (수확 후) | 씨앗, 비료, 방제약 |
| 사과 | 3~4월 (전정/시비) | 10~11월 (수확 후) | 비료, 농약, 봉지 |
| 감귤 | 3~4월 (시비) | 11~12월 (수확 후) | 비료, 방제약 |

이를 월별로 합산하면 아래와 같은 트래픽 패턴이 나온다.

```
1월  ▂▂ (휴경, 낮음)
2월  ▃▃ (다음 시즌 준비 조회)
3월  ██ (고추·사과·감귤 농자재 구매 최고점)
4월  ▇▇ (쌀·고추 파종 마무리)
5월  ▅▅ (쌀 모내기 + 양파 수확 후 상환)
6월  ▄▄ (양파 상환 + 생육기 추가 비료)
7월  ▃▃ (장마철, 낮음 + 불규칙)
8월  ▄▄ (고추 수확 후 상환 + 병충해 방제)
9월  ▄▄ (쌀·고추 상환 + 양파 파종 구매)
10월 ▃▃ (쌀·사과 상환)
11월 ▃▃ (사과·감귤 상환)
12월 ▂▂ (감귤 상환, 휴경)
```

> 연중 최고점은 **3월 (고추·사과·감귤 동시 파종 준비)**

---

#### 주간 패턴 (Weekly)

| 요일 | 트래픽 수준 | 이유 |
|------|-------------|------|
| 월~금 | 높음 (100%) | 농부 활동 집중 |
| 토 | 중간 (70%) | 주말 거래 |
| 일 | 낮음 (40%) | 휴식 |

---

#### 일간 패턴 (Daily)

| 시간대 | 트래픽 수준 | 이유 |
|--------|-------------|------|
| 06~09시 | 높음 (90%) | 농작업 시작 전 주문 확인 |
| 09~12시 | 중간 (60%) | 오전 구매 |
| 12~14시 | 낮음 (30%) | 점심 |
| 14~18시 | 중간 (55%) | 오후 구매 |
| 18~21시 | 높음 (85%) | 저녁 구매/상환 조회 피크 |
| 21~06시 | 최저 (10%) | 야간 |

---

#### 기상 패턴 (Regressor)

이벤트성 패턴은 제외하고, **기상 조건만 regressor로 반영**한다.

| 기상 조건 | 기간 | 영향 | 반영 방식 |
|-----------|------|------|-----------|
| 장마철 | 6월 말~7월 말 | 트래픽 ±20% 불규칙 변동 | `is_monsoon` regressor (0/1) |
| 태풍 시즌 | 8~9월 | 태풍 상륙 시 급감 후 급증 | `typhoon_index` regressor (0.0~1.0) |

```
장마철: 농작업 중단 → 플랫폼 접속 감소
태풍:   상륙 당일 급감 → 이후 피해 복구 농자재 구매로 급증
```

---

#### 이상치 (RCA 테스트용)

```
의도적으로 임계값 초과 구간 삽입 (연 3~4회)

시나리오 1: 파종기 트래픽 급증
- 3월 중 요청수가 예측값 대비 40% 초과
- CPU 85% 이상 지속 (30분)
- 원인: 고추·사과·감귤 파종 시즌 동시 집중

시나리오 2: 태풍 직후 급증
- 태풍 상륙 다음날 요청수 급증
- 예측 범위 상한 초과
- 원인: 피해 복구 농자재 긴급 구매

시나리오 3: 장마철 불규칙
- 7월 중 트래픽 패턴 불규칙
- 예측 편차 지속 (MAPE > 15%)
- 원인: 장마 영향으로 농작업 불규칙

→ RCA 파이프라인 테스트 및 Drift 감지 검증에 활용
```

---

### 1-4. 출력 데이터 형식

```python
# Prophet 학습용 형식
{
  "ds": "2024-03-15 09:00:00",  # 타임스탬프 (1시간 단위)
  "y": 85.3,                     # 요청수: 건/분 기준 평균값 (해당 1시간의 평균 RPS)
                                  # CPU: % (0~100)
  "is_monsoon": 0,               # 장마 여부 (0/1)
  "typhoon_index": 0.0,          # 태풍 영향 지수 (0.0~1.0)
}

# 파일 구조
data/
├── dummy_request_rate.csv       # 요청수 데이터 (2년치, 단위: 건/분)
├── dummy_cpu_utilization.csv    # CPU 사용률 데이터 (2년치, 단위: %)
└── dummy_anomaly_events.csv     # 이상치 구간 정보 (시나리오별)
```

---

### 1-5. 기술 스택

| 라이브러리 | 용도 |
|-----------|------|
| pandas | 데이터프레임 생성/저장 |
| numpy | 노이즈, 랜덤 패턴 생성 |

---

### 1-6. 주요 함수

```python
def generate_base_pattern(
    start_date: str,       # 시작일 "YYYY-MM-DD"
    end_date: str,         # 종료일 "YYYY-MM-DD"
    freq: str = "1H",      # 시간 단위 (1시간)
) -> pd.DataFrame:
    """기본 시계열 뼈대 생성 (타임스탬프 인덱스)"""

def apply_crop_seasonal_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """
    작물별 계절 패턴 적용
    - 쌀/고추/사과/감귤: 3~4월 파종 준비 피크
    - 양파: 9~10월 파종 피크
    - 수확 후 BNPL 상환 조회 패턴 포함
    """

def apply_weekly_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """요일별 패턴 적용 (월~금 높음, 주말 낮음)"""

def apply_daily_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """시간대별 패턴 적용 (오전 6~9시, 저녁 18~21시 피크)"""

def apply_weather_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """
    기상 패턴 적용 + regressor 컬럼 생성
    - is_monsoon: 6월 말~7월 말 (0/1)
    - typhoon_index: 8~9월 태풍 영향 지수 (0.0~1.0)
    """

def inject_anomalies(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """
    RCA 테스트용 이상치 구간 삽입
    - 시나리오 1: 파종기 트래픽 급증 (3월)
    - 시나리오 2: 태풍 직후 급증 (8~9월)
    - 시나리오 3: 장마철 불규칙 (7월)
    """

def save_to_csv(df: pd.DataFrame, path: str) -> None:
    """Prophet 형식으로 CSV 저장"""
```

---

### 1-7. 에러 처리

| 상황 | 처리 방법 |
|------|-----------|
| 날짜 형식 오류 | ValueError 발생 + 올바른 형식 안내 |
| 출력 경로 없음 | 디렉토리 자동 생성 |
| 데이터 크기 초과 | 기간 제한 경고 메시지 출력 |

---

### 1-8. 완료 기준

- [ ] 2년치 요청수, CPU 데이터 생성 가능
- [ ] 작물별 계절 패턴(쌀/사과/고추/감귤/양파)이 월별 트래픽에 반영됨
- [ ] 장마철 불규칙 패턴, 태풍 급감/급증 패턴 시각적으로 확인 가능
- [ ] 이상치 시나리오 3가지 구간이 CSV에 명확히 표시됨
- [ ] Prophet 학습에 바로 사용 가능한 형식으로 출력

---
---

## 모듈 2: Prophet 학습/예측 모듈

### 2-1. 개요

| 항목 | 내용 |
|------|------|
| 목적 | 농업 서비스 특화 시계열 예측 모델 학습 및 예측값 생성 |
| 입력 | 더미 데이터 CSV (또는 Prometheus API 데이터) |
| 출력 | 예측값 (Prometheus Custom Metric Push), MLflow 모델 등록 |
| 실행 방식 | 최초 학습: 수동 / 예측: K8s CronJob (매시간) |

---

### 2-2. 모델 구성

| 항목 | 모델 1 | 모델 2 |
|------|--------|--------|
| 이름 | request-rate-forecast | cpu-forecast |
| 예측 대상 | 요청수 (건/분, 1시간 평균) | CPU 사용률 (%) |
| 활용 환경 | On-Prem (Pod 스케일링) | Cloud (노드 스케일링) |
| 예측 horizon | 24시간 | 24시간 |
| Regressor | is_monsoon, typhoon_index | is_monsoon, typhoon_index |

> 두 모델 모두 장마/태풍 영향을 받으므로 동일한 regressor 적용

---

### 2-3. Prophet 모델 설정

```python
# 공통 설정
Prophet(
    yearly_seasonality=True,           # 작물별 계절 패턴 반영
    weekly_seasonality=True,           # 요일 패턴
    daily_seasonality=True,            # 시간대 패턴
    seasonality_mode='multiplicative', # 파종기/휴경기 진폭 차이 반영
    changepoint_prior_scale=0.05,      # 트렌드 변화점 민감도
    seasonality_prior_scale=10.0,      # 계절성 강도
)

# 커스텀 추가
.add_country_holidays(country_name='KR')   # 한국 공휴일 (자동 반영)
.add_regressor('is_monsoon')               # 장마 여부 (0/1)
.add_regressor('typhoon_index')            # 태풍 영향 지수 (0.0~1.0)
```

---

### 2-4. 예측값 활용처

| 활용처 | 방식 | 우선순위 |
|--------|------|----------|
| KEDA Custom Metric Push | 예측값을 Prometheus에 Push → KEDA가 감지 → Pod/노드 선제 조정 | 필수 |
| MAPE 계산 | 예측값 vs 실제값 비교 → 모델 성능 지표 | 필수 |
| RCA 편차 근거 | 이상 발생 시 "예측 N% vs 실제 N%" 근거로 활용 | 필수 |
| Headroom 계산 | 예측 피크값 기준 On-Prem 여유분 자동 계산 | 필수 |
| 이상치 탐지 보조 | 예측 상한/하한 벗어난 실제값 → 이상 징후 감지 | 있으면 좋음 |
| 리포트 피크 표시 | "이번 주 예상 피크: 수요일 오후 8시" | 있으면 좋음 |

---

### 2-5. 데이터 흐름

```
[학습]
더미 CSV (또는 Prometheus API)
        ↓
데이터 전처리 (Prophet 형식 변환)
        ↓
Prophet 학습 (파라미터 추정)
        ↓
교차검증 → MAPE 계산
        ↓
MLflow Model Registry 등록
        ↓
MinIO 모델 아티팩트 저장

[예측 - 매시간 CronJob]
MLflow에서 Production 모델 로드
        ↓
향후 24시간 예측값 생성
        ↓
[모델 1: On-Prem]            [모델 2: Cloud]
예측 RPS                      예측 CPU 사용률
        ↓                             ↓
RPS → 필요 Pod 수 변환        CPU → 필요 노드 수 추정
(calculate_required_pods)     (calculate_required_nodes)
        ↓                             ↓
Prometheus Custom Metric Push (KEDA 연동)
        ↓
PostgreSQL 저장 (RCA 편차 근거용)
        ↓
Headroom 계산 → 결과 저장
```

---

### 2-6. MLflow 관리

| 항목 | 내용 |
|------|------|
| Experiment | `aiops-request-forecast`, `aiops-cpu-forecast` |
| 기록 항목 | MAPE, changepoint_prior_scale, seasonality_prior_scale, 학습 데이터 기간 |
| 모델 버전 | Staging → (검증 통과) → Production → (교체 후) → Archived |
| 검증 기준 | MAPE < 0.15 |

---

### 2-7. RPS → Pod 수 변환 로직 (On-Prem)

```python
import math

def calculate_required_pods(
    predicted_rps: float,          # 예측 요청수 (건/분)
    capacity_per_pod: float,       # Pod 1개당 처리 가능한 요청수 (건/분)
                                   # PoC: 임의 설정, 실제 인프라 연결 시 실측값으로 교체
    safety_margin: float = 0.2,    # 안전 여유분 20%
) -> dict:
    """
    예측 RPS → 필요 Pod 수 변환
    
    예시:
      predicted_rps = 350 건/분
      capacity_per_pod = 100 건/분
      safety_margin = 0.2
      
      effective_capacity = 100 * (1 - 0.2) = 80 건/분
      required_pods = ceil(350 / 80) = 5개
    """
    effective_capacity = capacity_per_pod * (1 - safety_margin)
    required_pods = math.ceil(predicted_rps / effective_capacity)

    return {
        "predicted_rps": predicted_rps,
        "capacity_per_pod": capacity_per_pod,
        "effective_capacity": effective_capacity,
        "required_pods": required_pods,
    }


def calculate_required_nodes(
    predicted_cpu_pct: float,      # 예측 CPU 사용률 (%)
    cpu_per_node: float,           # 노드 1개당 CPU 코어 수
    cpu_per_pod: float,            # Pod 1개당 CPU 요청량 (cores)
    current_pods: int,             # 현재 Pod 수
    safety_margin: float = 0.2,
) -> dict:
    """
    예측 CPU 사용률 → 필요 노드 수 추정 (Cloud)
    
    예시:
      predicted_cpu_pct = 75%
      cpu_per_node = 4 cores
      cpu_per_pod = 0.5 cores
      current_pods = 10
      
      required_cpu = current_pods * cpu_per_pod * (predicted_cpu_pct / 100)
                   = 10 * 0.5 * 0.75 = 3.75 cores
      required_nodes = ceil(3.75 / (4 * 0.8)) = ceil(3.75 / 3.2) = 2개
    """
    required_cpu = current_pods * cpu_per_pod * (predicted_cpu_pct / 100)
    effective_cpu_per_node = cpu_per_node * (1 - safety_margin)
    required_nodes = math.ceil(required_cpu / effective_cpu_per_node)

    return {
        "predicted_cpu_pct": predicted_cpu_pct,
        "required_cpu_cores": required_cpu,
        "required_nodes": required_nodes,
    }
```

> **PoC 단계**: `capacity_per_pod`, `cpu_per_node`, `cpu_per_pod` 는 임의 설정값 사용
> **실제 인프라 연결 시**: 부하 테스트 실측값으로 교체

---

### 2-8. Headroom 계산 로직 (On-Prem)

```python
def calculate_headroom(
    predicted_peak_rps: float,     # 예측 피크 요청수 (건/분)
    total_cluster_capacity: float, # 클러스터 전체 처리 가능 요청수 (건/분)
    safety_margin: float = 0.2,    # 안전 여유분 20%
) -> dict:
    """
    On-Prem 기준 여유분 계산
    예측 피크가 전체 용량의 80%를 넘으면 경고
    """
    headroom_ratio = (total_cluster_capacity - predicted_peak_rps) / total_cluster_capacity

    return {
        "predicted_peak_rps": predicted_peak_rps,
        "total_cluster_capacity": total_cluster_capacity,
        "headroom_ratio": round(headroom_ratio, 3),
        "is_risk": headroom_ratio < safety_margin,
        "recommendation": "Pod 비율 조정 필요" if headroom_ratio < safety_margin else "정상"
    }
```

---

### 2-9. 기술 스택

| 라이브러리 | 용도 |
|-----------|------|
| prophet | 시계열 예측 모델 |
| mlflow | 실험 추적, 모델 버전 관리 |
| pandas | 데이터 전처리 |
| scipy | 교차검증 보조 |
| prometheus_client | Custom Metric Push |
| boto3 | MinIO/S3 모델 아티팩트 저장 |

---

### 2-10. 주요 함수

```python
def load_data(source: str = "csv", days: int = 60) -> pd.DataFrame:
    """
    source: "csv" (더미) | "prometheus" (실제)
    Prometheus 연결 시 source만 바꾸면 됨
    """

def train(df: pd.DataFrame, model_name: str) -> tuple[Prophet, float]:
    """학습 + MAPE 반환"""

def predict(model: Prophet, horizon_hours: int = 24) -> pd.DataFrame:
    """향후 N시간 예측값 생성"""

def calculate_required_pods(predicted_rps: float, capacity_per_pod: float, safety_margin: float = 0.2) -> dict:
    """예측 RPS → 필요 Pod 수 변환 (On-Prem)"""

def calculate_required_nodes(predicted_cpu_pct: float, cpu_per_node: float, cpu_per_pod: float, current_pods: int, safety_margin: float = 0.2) -> dict:
    """예측 CPU → 필요 노드 수 추정 (Cloud)"""

def calculate_headroom(predicted_peak_rps: float, total_cluster_capacity: float, safety_margin: float = 0.2) -> dict:
    """On-Prem 여유분 계산"""

def push_to_prometheus(predictions: pd.DataFrame, metric_name: str) -> None:
    """예측값 + 필요 Pod/노드 수 Prometheus Custom Metric으로 Push"""

def register_model(model: Prophet, model_name: str, mape: float) -> str:
    """MLflow Model Registry 등록, 버전 반환"""

def evaluate(model: Prophet, df: pd.DataFrame) -> float:
    """교차검증으로 MAPE 계산"""
```

---

### 2-11. 에러 처리

| 상황 | 처리 방법 |
|------|-----------|
| MAPE ≥ 0.15 | MLflow Staging 유지, Slack 경고 알림 |
| Prometheus Push 실패 | 로컬 파일로 fallback 저장, 재시도 3회 |
| MLflow 연결 실패 | 로컬 경로에 모델 저장 후 알림 |
| 데이터 부족 (< 30일) | 학습 중단 + 에러 로그 |
| capacity_per_pod 미설정 | 기본값 사용 + 경고 로그 (실측값 설정 권고) |

---

### 2-12. 완료 기준

- [ ] 더미 데이터로 모델 2개 학습 가능
- [ ] MAPE < 0.15 달성
- [ ] MLflow에 버전 관리 확인
- [ ] 예측 RPS → 필요 Pod 수 변환 정상 동작 (On-Prem)
- [ ] 예측 CPU → 필요 노드 수 추정 정상 동작 (Cloud)
- [ ] Prometheus Custom Metric Push 확인 (Pod 수 / 노드 수 포함)
- [ ] Headroom 계산값 정상 출력
- [ ] source="prometheus" 로 교체만 하면 실제 데이터 연동 가능한 구조

---

### 2-13. 연간 예측 (Annual Forecast)

#### 목적

24시간 예측만으로는 계절성에 따른 Pod 스케일링 변화를 직관적으로 확인하기 어렵다.
연간 예측(8,760시간)을 생성하여 **1월(최저) → 3월(피크) → 12월(최저)** 흐름을 한눈에 시각화한다.

#### Pod 수 설계 기준

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

#### 예상 월별 Pod 수

```
1월  ▂  1~2 pods  (휴경, 야간 최저)
2월  ▃  2~3 pods  (다음 시즌 준비)
3월  ██  6~8 pods  (고추·사과·감귤 파종 피크 + 이상치 +40%)
4월  ▇  5~7 pods  (쌀·고추 파종 마무리)
5월  ▅  3~5 pods  (모내기 + 상환 조회)
6월  ▄  3~4 pods  (장마 진입)
7월  ▃  2~3 pods  (장마 불규칙)
8월  ▄  3~4 pods  (수확 후 상환)
9월  ▄  3~5 pods  (태풍 이상치 +50% 반영)
10월 ▃  2~3 pods  (수확 후 상환)
11월 ▃  2~3 pods  (감귤 상환)
12월 ▂  1~2 pods  (휴경)
```

#### 출력

| 파일 | 내용 |
|------|------|
| `data/predictions_annual_2025.csv` | 8,760행. 컬럼: `ds`, `yhat`, `yhat_lower`, `yhat_upper`, `required_pods` |
| `data/plot_annual_pods.png` | 월별 Pod 수 추이 + 피크 구간 음영 시각화 |

#### 주요 함수

```python
def predict_annual(
    model: Prophet,
    df: pd.DataFrame,
    year: int = 2025,
) -> pd.DataFrame:
    """
    지정 연도 전체(1월 1일 ~ 12월 31일) 시간별 예측값 생성
    - regressor(is_monsoon, typhoon_index)는 학습 데이터 동월 평균으로 추정
    - required_pods: MIN_PODS(1) ~ MAX_PODS(8) clamp 적용
    """
```

#### 실행

```bash
# 학습 후 연간 예측까지 한 번에
python src/train_predict.py --annual

# 시각화
python src/compare_plot.py
```

#### 완료 기준

- [ ] 8,760행 연간 예측 CSV 생성
- [ ] required_pods 범위가 1~8 내에 존재
- [ ] 3월 피크 구간에서 MAX_PODS(8) 도달 확인
- [ ] plot_annual_pods.png 월별 Pod 추이 시각적으로 확인 가능

---

### 2-14. compare_plot.py 변경 사항 (연간 예측 시각화 반영)

#### 추가된 상수

```python
MIN_PODS = 1   # train_predict.py 와 동기화
MAX_PODS = 8
```

#### 추가된 함수: `plot_annual_pods(annual: pd.DataFrame)`

| 항목 | 내용 |
|------|------|
| 입력 | `predictions_annual_2025.csv` |
| 출력 | `data/plot_annual_pods.png` |
| 구성 | 2단 그래프 |

```
상단: 예측 RPS (일별 평균) + 신뢰구간 음영
      - 3월 (파종 피크) 빨간 음영
      - 장마(6/25~7/25) 하늘색 음영
      - 태풍 시즌(8~9월) 주황 음영

하단: 필요 Pod 수 (일별 최대, step 그래프)
      - MIN_PODS(1) / MAX_PODS(8) 기준선 표시
      - 3월 피크 구간 빨간 음영
```

#### `main()` 변경 사항

```python
# 추가된 로드
annual = load(DATA_DIR / "predictions_annual_2025.csv", "연간 예측")

# 추가된 호출
if annual is not None:
    plot_annual_pods(annual)

# 출력 안내 메시지 추가
print("  data/plot_annual_pods.png — 2025년 연간 Pod 수 추이")
```

> `predictions_annual_2025.csv` 가 없으면 경고만 출력하고 나머지 그래프는 정상 생성됨.
