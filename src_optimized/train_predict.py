"""
최적화 버전: Prophet 학습/예측 (로그 변환 적용)

baseline(src/train_predict.py) 대비 변경점:
  - 학습 전 y에 log1p 변환 적용
  - 예측 후 expm1 역변환으로 실제 단위 복원
  - 극단적 범위(0.8~135 건/분, 169배)를 균등하게 압축해 계절성 학습 개선

실행: python src_optimized/train_predict.py
출력: models/request-rate-forecast-log.pkl
      models/runs.json  (기록 추가)
      data/predictions_log_request_rate.csv
      data/predictions_log_annual_2025.csv
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

ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

CAPACITY_PER_POD       = 21.1
SAFETY_MARGIN          = 0.2
MIN_PODS               = 1
MAX_PODS               = 8
TOTAL_CLUSTER_CAPACITY = CAPACITY_PER_POD * MAX_PODS
MAPE_THRESHOLD         = 0.15
MODEL_NAME             = "request-rate-forecast-log"


# ── 로그 변환 ──────────────────────────────────────────────────

def log_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["y"] = np.log1p(df["y"])
    return df


def inverse_transform(forecast: pd.DataFrame) -> pd.DataFrame:
    fc = forecast.copy()
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        if col in fc.columns:
            fc[col] = np.expm1(fc[col])
    return fc


# ── 데이터 ────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    path = DATA_DIR / "dummy_request_rate.csv"
    df = pd.read_csv(path, parse_dates=["ds"])
    print(f"[데이터] {len(df):,}행  ({df['ds'].min().date()} ~ {df['ds'].max().date()})")
    return df


# ── 모델 ──────────────────────────────────────────────────────

def build_model() -> Prophet:
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode="additive",      # log 변환 후 additive가 적합
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
    )
    m.add_country_holidays(country_name="KR")
    m.add_regressor("is_monsoon")
    m.add_regressor("typhoon_index")
    return m


def train(df: pd.DataFrame) -> tuple[Prophet, float]:
    df_log = log_transform(df)
    print(f"\n[학습] {MODEL_NAME} (log 변환 적용)")
    m = build_model()
    m.fit(df_log)
    print("[학습] 완료")

    print("[평가] 교차검증 중...")
    df_cv = cross_validation(
        m, initial="365 days", period="30 days", horizon="7 days", disable_tqdm=True
    )
    df_perf = performance_metrics(df_cv, rolling_window=1)

    for col in ["mape", "smape", "mdape"]:
        if col in df_perf.columns:
            mape = df_perf[col].mean()
            print(f"[평가] {col.upper()} = {mape:.4f}  ({'통과' if mape < MAPE_THRESHOLD else '미달'})")
            return m, mape

    raise ValueError("성능 지표 컬럼 없음")


# ── 예측 ──────────────────────────────────────────────────────

def predict_24h(model: Prophet, df: pd.DataFrame) -> pd.DataFrame:
    last_ds = df["ds"].max()
    future_dates = pd.date_range(
        start=last_ds + pd.Timedelta(hours=1), periods=24, freq="1h"
    )
    future = pd.DataFrame({"ds": future_dates})

    last_window = df[df["ds"] >= last_ds - pd.Timedelta(hours=24)]
    future["is_monsoon"]    = int(last_window["is_monsoon"].mean().round())
    future["typhoon_index"] = last_window["typhoon_index"].mean()

    forecast = model.predict(future)
    forecast = inverse_transform(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]])
    forecast["required_pods"] = forecast["yhat"].apply(_to_pods)

    print(f"[예측] 24시간  ({future_dates[0].date()} ~ {future_dates[-1].date()})")
    return forecast


def predict_annual(model: Prophet, year: int = 2025) -> pd.DataFrame:
    dates  = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31 23:00", freq="1h")
    future = pd.DataFrame({"ds": dates})

    future["is_monsoon"] = (
        (future["ds"] >= f"{year}-06-25") & (future["ds"] <= f"{year}-07-25")
    ).astype(int)

    future["typhoon_index"] = 0.0
    rng     = np.random.default_rng(year)
    month   = int(rng.integers(8, 10))
    day     = int(rng.integers(1, 26))
    landfall = pd.Timestamp(f"{year}-{month:02d}-{day:02d}")
    for offset, idx_val in [(-1, 0.1), (0, 0.9), (1, 0.7), (2, 0.3), (3, 0.1)]:
        target = landfall + pd.Timedelta(days=offset)
        mask   = future["ds"].dt.date == target.date()
        future.loc[mask, "typhoon_index"] = np.maximum(
            future.loc[mask, "typhoon_index"].values, idx_val
        )

    forecast = model.predict(future)
    forecast = inverse_transform(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]])
    forecast["required_pods"] = forecast["yhat"].apply(_to_pods)

    out = DATA_DIR / f"predictions_log_annual_{year}.csv"
    forecast.to_csv(out, index=False)
    print(f"[연간 예측] {year}년  Pod 범위: {forecast['required_pods'].min()}~{forecast['required_pods'].max()}개")
    print(f"[저장] {out.name}")
    return forecast


def _to_pods(rps: float) -> int:
    eff = CAPACITY_PER_POD * (1 - SAFETY_MARGIN)
    return max(MIN_PODS, min(math.ceil(max(rps, 0) / eff), MAX_PODS))


# ── 저장 ──────────────────────────────────────────────────────

def save_model(model: Prophet, mape: float) -> None:
    path = MODEL_DIR / f"{MODEL_NAME}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)

    runs_path = MODEL_DIR / "runs.json"
    runs = json.loads(runs_path.read_text()) if runs_path.exists() else []
    runs.append({
        "model_name": MODEL_NAME,
        "trained_at": datetime.now().isoformat(),
        "mape":  round(mape, 6),
        "stage": "Production" if mape < MAPE_THRESHOLD else "Staging",
        "transform": "log1p",
        "path": str(path),
    })
    runs_path.write_text(json.dumps(runs, indent=2, ensure_ascii=False))

    stage = "Production" if mape < MAPE_THRESHOLD else "Staging"
    print(f"[저장] {path.name}  stage={stage}")


# ── 메인 ──────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  최적화 모델: Prophet + log 변환  (request_rate)")
    print("=" * 55)

    df = load_data()

    # 학습
    model, mape = train(df)
    save_model(model, mape)

    # 24시간 예측
    forecast_24h = predict_24h(model, df)
    out_24h = DATA_DIR / "predictions_log_request_rate.csv"
    forecast_24h.to_csv(out_24h, index=False)
    print(f"[저장] {out_24h.name}  (24 rows)")

    # 연간 예측
    print("\n" + "=" * 55)
    print("  연간 예측 (2025년)")
    print("=" * 55)
    predict_annual(model, year=2025)

    # 결과 요약
    peak_rps = forecast_24h["yhat"].max()
    headroom = (TOTAL_CLUSTER_CAPACITY - peak_rps) / TOTAL_CLUSTER_CAPACITY

    print("\n" + "=" * 55)
    print("  예측 결과 요약 (향후 24시간)")
    print("=" * 55)
    print(f"  예측 RPS 범위: {forecast_24h['yhat'].min():.1f} ~ {forecast_24h['yhat'].max():.1f} 건/분")
    print(f"  필요 Pod 범위: {forecast_24h['required_pods'].min()} ~ {forecast_24h['required_pods'].max()} 개")
    print(f"  피크 시각:     {forecast_24h.loc[forecast_24h['yhat'].idxmax(), 'ds']}")
    print(f"  Headroom:      {headroom:.1%}  ({'✅ 정상' if headroom >= SAFETY_MARGIN else '⚠️  위험'})")
    print(f"  MAPE:          {mape:.4f}  ({'✅ 통과' if mape < MAPE_THRESHOLD else '❌ 미달'})")
    print("=" * 55)


if __name__ == "__main__":
    main()
