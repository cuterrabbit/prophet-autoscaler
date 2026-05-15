# 모듈 3: 재학습 파이프라인 모듈

---

## 3-1. 개요

| 항목 | 내용 |
|------|------|
| 목적 | Prophet 모델을 최신 데이터로 재학습하여 예측 정확도 유지 |
| 입력 | 최근 2년치 데이터 (Prometheus 60일 + MinIO/S3 이전 데이터) |
| 출력 | 새 모델 버전 MLflow 등록, Blue/Green 교체 |
| 실행 방식 | 연 1회 K8s CronJob (1~2월 농한기) |

---

## 3-2. 재학습 전략

### 왜 연 1회인가?

```
Prophet은 학습 데이터가 동일하면 재학습해도 결과가 거의 동일함.
→ 의미있는 재학습은 "새 데이터가 충분히 쌓였을 때"만 가능.
→ 농업 서비스는 1년 단위 계절성이 핵심이므로,
  1년치 새 데이터가 쌓인 시점(농한기)에 재학습하는 것이 최적.
```

### 재학습 데이터 구성

```
Prometheus (최근 60일)
        +
MinIO/S3 (그 이전 데이터)
        ↓
합산 → 최근 2년치 로드
        ↓
재학습 실행
```

> **왜 2년치인가?**
> Prophet이 yearly seasonality를 안정적으로 학습하려면 최소 2년치 필요.
> 1년치만 있으면 "매년 반복되는 패턴"인지 확신할 수 없음.

### 모니터링 지표 (재학습 트리거 아님)

MAE, KS-test는 **재학습 트리거가 아니라 성능 모니터링 + 경고 용도**로만 사용한다.
같은 데이터로 재학습해도 결과가 달라지지 않기 때문이다.

| 지표 | 감지 조건 | 행동 |
|------|-----------|------|
| MAE | Pod 수 오차 N개 이상 지속 | Slack 경고만 (재학습 X) |
| KS-test | 작년 동월 vs 올해 동월 p-value < 0.05 | Slack 경고만 (재학습 X) |

---

## 3-3. 데이터 저장 전략

재학습을 위해 Prometheus 60일 이전 데이터를 **MinIO/S3에 별도 보관**해야 한다.

```
[매일 CronJob]
Prometheus에서 전일 데이터 수집
        ↓
MinIO/S3에 Parquet 형식으로 적재
(파티셔닝: year/month/day)

data/
└── metrics/
    └── request_rate/
        └── year=2024/
            └── month=03/
                └── day=15/
                    └── data.parquet
```

| 저장소 | 보존 기간 | 용도 |
|--------|-----------|------|
| Prometheus | 60일 | 실시간 모니터링, 단기 RCA |
| MinIO/S3 | 3년 | 재학습용 장기 데이터 |

---

## 3-4. 재학습 흐름

```
[연 1회 CronJob - 1~2월 농한기]
        ↓
데이터 로드
├── Prometheus API → 최근 60일
└── MinIO/S3 → 그 이전 ~ 2년 전
        ↓
데이터 병합 + 전처리
(중복 제거, 결측값 처리, Prophet 형식 변환)
        ↓
모델 2개 재학습 (request-rate, cpu)
        ↓
교차검증 → MAE + MAPE 계산
        ↓
검증 통과? (MAPE < 0.15)
    ├── Yes → MLflow Staging 등록
    │          ↓
    │        Blue/Green 교체
    │        (Staging → Production, 기존 → Archived)
    │          ↓
    │        MinIO/S3 모델 아티팩트 저장
    │          ↓
    │        Slack 성공 알림
    │
    └── No  → MLflow Staging 유지 (Production 교체 안 함)
               ↓
             Slack 실패 알림 (MAPE 값 포함)
             수동 검토 요청
```

---

## 3-5. 모니터링 파이프라인 (별도 CronJob)

재학습과 별개로 **매일 모델 성능을 모니터링**한다.

```
[매일 CronJob]
Prometheus에서 전일 실제값 수집
        ↓
MLflow Production 모델의 전일 예측값 조회
        ↓
MAE 계산 (예측 Pod 수 vs 실제 필요 Pod 수)
KS-test (작년 동월 vs 올해 동월)
        ↓
이상 감지 시 Slack 경고
(재학습 트리거 X, 경고만)
```

### MAE 계산 방식

```python
def calculate_mae_pods(
    predicted_rps: float,
    actual_rps: float,
    capacity_per_pod: float,
) -> dict:
    """
    예측 RPS vs 실제 RPS를 Pod 수 오차로 변환
    
    예시:
      predicted_rps = 350, actual_rps = 420
      capacity_per_pod = 80
      
      predicted_pods = ceil(350 / 80) = 5
      actual_pods    = ceil(420 / 80) = 6
      pod_error = 6 - 5 = 1개 오차
    """
    predicted_pods = math.ceil(predicted_rps / capacity_per_pod)
    actual_pods = math.ceil(actual_rps / capacity_per_pod)
    pod_error = abs(actual_pods - predicted_pods)

    return {
        "predicted_pods": predicted_pods,
        "actual_pods": actual_pods,
        "pod_error": pod_error,
        "is_warning": pod_error >= WARNING_THRESHOLD,  # 기본값: 2개 이상 오차
    }
```

### KS-test 방식

```python
def detect_distribution_drift(
    reference_month: int,   # 비교 기준: 작년 동월
    current_month: int,     # 현재 월
    p_threshold: float = 0.05,
) -> dict:
    """
    작년 동월 vs 올해 동월 분포 비교
    (전체 기간 비교 X → 계절성으로 인한 오탐 방지)
    """
    reference_data = load_from_s3(year=current_year - 1, month=reference_month)
    current_data = load_from_prometheus(month=current_month)

    ks_stat, p_value = stats.ks_2samp(reference_data, current_data)

    return {
        "ks_statistic": ks_stat,
        "p_value": p_value,
        "is_warning": p_value < p_threshold,
        "message": "작년 동월 대비 분포 변화 감지" if p_value < p_threshold else "정상",
    }
```

---

## 3-6. Blue/Green 교체 로직

```python
def blue_green_swap(
    model_name: str,
    new_version: str,
    mape: float,
    mape_threshold: float = 0.15,
) -> bool:
    """
    검증 통과 시 Staging → Production 교체
    기존 Production → Archived
    """
    client = MlflowClient()

    if mape >= mape_threshold:
        notify_slack(f"재학습 실패: {model_name} MAPE={mape:.3f} (기준 {mape_threshold})")
        return False

    # 기존 Production → Archived
    current = client.get_latest_versions(model_name, stages=["Production"])
    if current:
        client.transition_model_version_stage(
            name=model_name,
            version=current[0].version,
            stage="Archived"
        )

    # 새 버전 Staging → Production
    client.transition_model_version_stage(
        name=model_name,
        version=new_version,
        stage="Production"
    )

    notify_slack(f"재학습 성공: {model_name} v{new_version} MAPE={mape:.3f}")
    return True
```

---

## 3-7. 기술 스택

| 라이브러리 | 용도 |
|-----------|------|
| prophet | 모델 재학습 |
| mlflow | 버전 관리, Blue/Green 교체 |
| pandas | 데이터 병합/전처리 |
| scipy | KS-test |
| boto3 | MinIO/S3 데이터 로드/저장 |
| prometheus_client | 실제값 수집 |

---

## 3-8. 주요 함수

```python
def load_training_data(years: int = 2) -> pd.DataFrame:
    """
    Prometheus (최근 60일) + MinIO/S3 (이전 데이터) 병합
    → 최근 N년치 반환
    """

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """중복 제거, 결측값 처리, Prophet 형식 변환"""

def retrain(df: pd.DataFrame, model_name: str) -> tuple[Prophet, float]:
    """재학습 + MAPE 반환"""

def blue_green_swap(model_name: str, new_version: str, mape: float) -> bool:
    """검증 통과 시 Production 교체"""

def save_daily_metrics(df: pd.DataFrame) -> None:
    """전일 데이터 MinIO/S3에 Parquet 저장"""

def calculate_mae_pods(predicted_rps: float, actual_rps: float, capacity_per_pod: float) -> dict:
    """예측 vs 실제 Pod 수 오차 계산"""

def detect_distribution_drift(reference_month: int, current_month: int) -> dict:
    """작년 동월 vs 올해 동월 KS-test"""

def notify_slack(message: str, level: str = "info") -> None:
    """Slack 알림 발송 (info / warning / error)"""
```

---

## 3-9. CronJob 구성

| CronJob | 주기 | 실행 스크립트 | 역할 |
|---------|------|---------------|------|
| daily-data-backup | 매일 새벽 2시 | save_daily_metrics.py | 전일 데이터 S3 적재 |
| daily-model-monitor | 매일 오전 8시 | monitor.py | MAE + KS-test 모니터링 |
| annual-retrain | 매년 1월 15일 | retrain.py | 연간 재학습 |

---

## 3-10. 에러 처리

| 상황 | 처리 방법 |
|------|-----------|
| S3 데이터 로드 실패 | Prometheus 60일치만으로 재학습 시도 + 경고 알림 |
| 재학습 MAPE ≥ 0.15 | Staging 유지, Slack 실패 알림 + 수동 검토 요청 |
| Blue/Green 교체 실패 | 기존 Production 유지, Slack 알림 |
| 일별 데이터 백업 실패 | 재시도 3회 후 Slack 경고 |

---

## 3-11. 완료 기준

- [ ] 연간 재학습 CronJob 정상 동작
- [ ] Prometheus + S3 데이터 병합 로드 정상 동작
- [ ] 재학습 후 MAPE < 0.15 달성 시 Blue/Green 교체 확인
- [ ] MAPE ≥ 0.15 시 교체 없이 Slack 실패 알림 발송 확인
- [ ] 매일 데이터 S3 백업 CronJob 정상 동작
- [ ] MAE Pod 오차 경고 Slack 발송 확인
- [ ] KS-test 작년 동월 비교 정상 동작 확인
