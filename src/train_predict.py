"""
모듈 2: Prophet 학습/예측
참조: module-spec/module-spec-1-2.md

실행: python src/train_predict.py
출력: models/request_rate_model.pkl       (학습된 모델)
      models/runs.json                    (MLflow 대체 - 실험 기록)
      data/predictions_request_rate.csv   (24시간 예측값 + Pod 수)
"""
import json
import math
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ── 설정 ──────────────────────────────────────────────────────
CAPACITY_PER_POD = 21.1   # Pod 1개당 처리 가능 요청수 (건/분): 피크(135) / MAX_PODS(8) / (1-margin)
SAFETY_MARGIN    = 0.2    # 안전 여유분 20%
MIN_PODS = 1
MAX_PODS = 8
TOTAL_CLUSTER_CAPACITY = CAPACITY_PER_POD * MAX_PODS  # 168.8 건/분
MAPE_THRESHOLD   = 0.15


# ── 데이터 로드 ───────────────────────────────────────────────

def load_data(source: str = "csv", days: int = 60) -> pd.DataFrame:
    """
    source: "csv" (더미) | "prometheus" (실제, 추후 교체)
    """
    if source == "csv":
        path = DATA_DIR / "dummy_request_rate.csv"
        df = pd.read_csv(path, parse_dates=["ds"])
        print(f"[데이터] CSV 로드 완료: {len(df):,} rows ({df['ds'].min().date()} ~ {df['ds'].max().date()})")
        return df
    raise NotImplementedError("source='prometheus' 는 실제 인프라 연결 후 구현")


# ── 학습 ──────────────────────────────────────────────────────

def build_model() -> Prophet:
    """스펙 기반 Prophet 모델 생성"""
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
    )
    m.add_country_holidays(country_name="KR")
    m.add_regressor("is_monsoon")
    m.add_regressor("typhoon_index")
    return m


def train(df: pd.DataFrame, model_name: str) -> tuple[Prophet, float]:
    """학습 + MAPE 반환"""
    print(f"\n[학습] {model_name} 모델 학습 시작...")
    m = build_model()
    m.fit(df)
    print(f"[학습] 완료")

    mape = evaluate(m, df)
    return m, mape


def evaluate(model: Prophet, df: pd.DataFrame) -> float:
    """교차검증으로 MAPE 계산 (prophet 신버전: smape 사용)"""
    print("[평가] 교차검증 중...")
    # yearly seasonality 안정 학습을 위해 initial=365일
    df_cv = cross_validation(
        model,
        initial="365 days",
        period="30 days",
        horizon="7 days",
        disable_tqdm=True,
    )
    df_perf = performance_metrics(df_cv, rolling_window=1)

    # prophet 버전에 따라 컬럼명 다름: mape > smape > mdape 순으로 시도
    for col in ["mape", "smape", "mdape"]:
        if col in df_perf.columns:
            mape = df_perf[col].mean()
            print(f"[평가] {col.upper()} = {mape:.4f} ({'통과' if mape < MAPE_THRESHOLD else '미달'})")
            return mape
    raise ValueError("성능 지표 컬럼을 찾을 수 없음")


# ── 예측 ──────────────────────────────────────────────────────

def predict(model: Prophet, df: pd.DataFrame, horizon_hours: int = 24) -> pd.DataFrame:
    """
    향후 N시간 예측값 생성
    - 마지막 학습 시점 이후 24시간
    - regressor(is_monsoon, typhoon_index)는 마지막 관측값으로 채움
    """
    last_ds = df["ds"].max()
    future_dates = pd.date_range(
        start=last_ds + pd.Timedelta(hours=1),
        periods=horizon_hours,
        freq="1h",
    )
    future = pd.DataFrame({"ds": future_dates})

    # regressor: 마지막 24시간 평균으로 미래 추정
    last_window = df[df["ds"] >= last_ds - pd.Timedelta(hours=24)]
    future["is_monsoon"]    = int(last_window["is_monsoon"].mean().round())
    future["typhoon_index"] = last_window["typhoon_index"].mean()

    forecast = model.predict(future)
    print(f"[예측] {horizon_hours}시간 예측 완료 ({future_dates[0].date()} ~ {future_dates[-1].date()})")
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


# ── 연간 예측 ─────────────────────────────────────────────────

def predict_annual(model: Prophet, year: int = 2025) -> pd.DataFrame:
    """
    지정 연도 전체(1월 1일 ~ 12월 31일) 시간별 예측값 생성
    - is_monsoon: 6/25 ~ 7/25 고정
    - typhoon_index: seed 고정 랜덤 태풍 1회 (재현성 보장)
    - required_pods: MIN_PODS ~ MAX_PODS clamp 적용
    """
    dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31 23:00", freq="1h")
    future = pd.DataFrame({"ds": dates})

    future["is_monsoon"] = (
        (future["ds"] >= f"{year}-06-25") & (future["ds"] <= f"{year}-07-25")
    ).astype(int)

    future["typhoon_index"] = 0.0
    rng = np.random.default_rng(year)
    month = int(rng.integers(8, 10))
    day   = int(rng.integers(1, 26))
    landfall = pd.Timestamp(f"{year}-{month:02d}-{day:02d}")
    for offset, idx_val in [(-1, 0.1), (0, 0.9), (1, 0.7), (2, 0.3), (3, 0.1)]:
        target = landfall + pd.Timedelta(days=offset)
        mask = future["ds"].dt.date == target.date()
        future.loc[mask, "typhoon_index"] = np.maximum(
            future.loc[mask, "typhoon_index"].values, idx_val
        )

    forecast = model.predict(future)
    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()

    eff = CAPACITY_PER_POD * (1 - SAFETY_MARGIN)
    result["required_pods"] = result["yhat"].apply(
        lambda rps: max(MIN_PODS, min(math.ceil(max(rps, 0) / eff), MAX_PODS))
    )

    out_path = DATA_DIR / f"predictions_annual_{year}.csv"
    result.to_csv(out_path, index=False)
    print(f"[연간 예측] {year}년 완료 ({len(result):,} rows)")
    print(f"  Pod 범위: {result['required_pods'].min()} ~ {result['required_pods'].max()} 개")
    print(f"  피크 시각: {result.loc[result['yhat'].idxmax(), 'ds']}")
    print(f"[저장] {out_path.name}")
    return result


# ── Pod/노드 수 계산 ──────────────────────────────────────────

def calculate_required_pods(
    predicted_rps: float,
    capacity_per_pod: float = CAPACITY_PER_POD,
    safety_margin: float = SAFETY_MARGIN,
) -> dict:
    """예측 RPS → 필요 Pod 수 변환 (On-Prem)"""
    effective_capacity = capacity_per_pod * (1 - safety_margin)
    required_pods = math.ceil(max(predicted_rps, 0) / effective_capacity)
    required_pods = max(MIN_PODS, min(required_pods, MAX_PODS))
    return {
        "predicted_rps": round(predicted_rps, 2),
        "capacity_per_pod": capacity_per_pod,
        "effective_capacity": effective_capacity,
        "required_pods": required_pods,
    }


def calculate_headroom(
    predicted_peak_rps: float,
    total_cluster_capacity: float = TOTAL_CLUSTER_CAPACITY,
    safety_margin: float = SAFETY_MARGIN,
) -> dict:
    """On-Prem 여유분 계산. headroom_ratio < safety_margin 이면 위험"""
    headroom_ratio = (total_cluster_capacity - predicted_peak_rps) / total_cluster_capacity
    return {
        "predicted_peak_rps": round(predicted_peak_rps, 2),
        "total_cluster_capacity": total_cluster_capacity,
        "headroom_ratio": round(headroom_ratio, 3),
        "is_risk": headroom_ratio < safety_margin,
        "recommendation": "Pod 비율 조정 필요" if headroom_ratio < safety_margin else "정상",
    }


# ── 저장 ──────────────────────────────────────────────────────

def save_model(model: Prophet, model_name: str, mape: float) -> str:
    """모델 .pkl 저장 + runs.json 기록 (MLflow 로컬 대체)"""
    path = MODEL_DIR / f"{model_name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)

    # 실험 기록
    runs_path = MODEL_DIR / "runs.json"
    runs = json.loads(runs_path.read_text()) if runs_path.exists() else []
    runs.append({
        "model_name": model_name,
        "trained_at": datetime.now().isoformat(),
        "mape": round(mape, 6),
        "stage": "Production" if mape < MAPE_THRESHOLD else "Staging",
        "path": str(path),
    })
    runs_path.write_text(json.dumps(runs, indent=2, ensure_ascii=False))

    stage = "Production" if mape < MAPE_THRESHOLD else "Staging"
    print(f"[저장] 모델: {path.name}  stage={stage}")
    return str(path)


def load_model(model_name: str) -> Prophet:
    """저장된 모델 로드"""
    path = MODEL_DIR / f"{model_name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def save_predictions(forecast: pd.DataFrame, pod_col: pd.Series, path: Path) -> None:
    """예측값 + Pod 수 CSV 저장 (Prometheus Push 로컬 대체)"""
    out = forecast.copy()
    out["required_pods"] = pod_col.values
    out.to_csv(path, index=False)
    print(f"[저장] 예측값: {path.name}  ({len(out)} rows)")


# ── 메인 ──────────────────────────────────────────────────────

def main():
    MODEL_NAME = "request-rate-forecast"

    print("=" * 50)
    print("  모듈 2: Prophet 학습/예측 (request_rate)")
    print("=" * 50)

    # 1. 데이터 로드
    df = load_data(source="csv")

    # 2. 학습
    model, mape = train(df, MODEL_NAME)

    # 3. 모델 저장
    save_model(model, MODEL_NAME, mape)

    # 4. 24시간 예측
    forecast = predict(model, df, horizon_hours=24)

    # 5. 예측값 → Pod 수 변환
    forecast["required_pods"] = forecast["yhat"].clip(lower=0).apply(
        lambda rps: calculate_required_pods(rps)["required_pods"]
    )

    # 6. Headroom 계산 (예측 피크 기준)
    peak_rps = forecast["yhat"].max()
    headroom = calculate_headroom(peak_rps)

    # 7. 예측 결과 저장
    out_path = DATA_DIR / "predictions_request_rate.csv"
    forecast.to_csv(out_path, index=False)
    print(f"[저장] 예측값: {out_path.name}  ({len(forecast)} rows)")

    # 8. 결과 요약
    print("\n" + "=" * 50)
    print("  예측 결과 요약 (향후 24시간)")
    print("=" * 50)
    print(f"  예측 시작:    {forecast['ds'].iloc[0]}")
    print(f"  예측 종료:    {forecast['ds'].iloc[-1]}")
    print(f"  예측 RPS 범위: {forecast['yhat'].min():.1f} ~ {forecast['yhat'].max():.1f} 건/분")
    print(f"  필요 Pod 범위: {forecast['required_pods'].min()} ~ {forecast['required_pods'].max()} 개")
    print(f"  피크 시각:    {forecast.loc[forecast['yhat'].idxmax(), 'ds']}")
    print(f"  피크 RPS:     {peak_rps:.1f} 건/분")
    print()
    print(f"  [Headroom]")
    print(f"    예측 피크 RPS:     {headroom['predicted_peak_rps']} 건/분")
    print(f"    클러스터 용량:     {headroom['total_cluster_capacity']} 건/분")
    print(f"    여유분(headroom):  {headroom['headroom_ratio']:.1%}")
    print(f"    상태:              {'⚠️  위험' if headroom['is_risk'] else '✅ 정상'} - {headroom['recommendation']}")
    print()
    print(f"  [모델 성능]")
    print(f"    MAPE:  {mape:.4f}  ({'✅ 기준 통과' if mape < MAPE_THRESHOLD else '❌ 기준 미달 (Staging 유지)'})")
    print("=" * 50)

    # Pod 수 시간별 출력
    print("\n  시간별 예측 (상위 5개 피크)")
    top5 = forecast.nlargest(5, "yhat")[["ds", "yhat", "required_pods"]]
    top5.columns = ["시각", "예측 RPS (건/분)", "필요 Pod 수"]
    print(top5.to_string(index=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prophet 학습/예측")
    parser.add_argument("--annual", action="store_true", help="2025년 연간 예측 추가 생성")
    args, _ = parser.parse_known_args()

    main()

    print("\n" + "=" * 50)
    print("  연간 예측 (2025년)")
    print("=" * 50)
    m = load_model("request-rate-forecast")
    predict_annual(m, year=2025)
