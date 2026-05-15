"""
Optuna 버전: Prophet 학습/예측 (로그 변환 + Optuna 최적 파라미터 + yhat_upper Pod)

src_tuned 대비 개선점:
  - optuna_search_results.csv 에서 최적 파라미터 3개 로드
    (changepoint_prior_scale, seasonality_prior_scale, holidays_prior_scale)
  - Pod 계산을 yhat → yhat_upper 기반으로 변경
    → 신뢰구간 상한 기준으로 Pod을 예약해 과소 프로비저닝 억제

실행: python src_optuna/train_predict.py
출력: models/request-rate-forecast-optuna.pkl
      models/runs.json  (기록 추가)
      data/predictions_optuna_request_rate.csv
      data/predictions_optuna_annual_2025.csv
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
MODEL_NAME             = "request-rate-forecast-optuna"

DEFAULT_CPS = 0.1
DEFAULT_SPS = 10.0
DEFAULT_HPS = 10.0


# ── 하이퍼파라미터 로드 ────────────────────────────────────────

def load_best_params() -> tuple[float, float, float]:
    path = DATA_DIR / "optuna_search_results.csv"
    if path.exists():
        df  = pd.read_csv(path)
        row = df.sort_values("smape").iloc[0]
        cps = float(row["changepoint_prior_scale"])
        sps = float(row["seasonality_prior_scale"])
        hps = float(row["holidays_prior_scale"])
        print(f"[파라미터] optuna_search_results.csv 로드")
        print(f"           cps={cps:.4f}  sps={sps:.4f}  hps={hps:.4f}")
    else:
        cps, sps, hps = DEFAULT_CPS, DEFAULT_SPS, DEFAULT_HPS
        print(f"[파라미터] 탐색 결과 없음 — 기본값 사용  cps={cps}  sps={sps}  hps={hps}")
        print("           (먼저 python src_optuna/hyperparameter_search.py 실행 권장)")
    return cps, sps, hps


# ── 변환 ──────────────────────────────────────────────────────

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

def build_model(cps: float, sps: float, hps: float) -> Prophet:
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode="additive",
        changepoint_prior_scale=cps,
        seasonality_prior_scale=sps,
        holidays_prior_scale=hps,
    )
    m.add_country_holidays(country_name="KR")
    m.add_regressor("is_monsoon")
    m.add_regressor("typhoon_index")
    return m


def train(df: pd.DataFrame, cps: float, sps: float, hps: float) -> tuple[Prophet, float]:
    df_log = log_transform(df)
    print(f"\n[학습] {MODEL_NAME}  (cps={cps:.4f}, sps={sps:.4f}, hps={hps:.4f})")
    m = build_model(cps, sps, hps)
    m.fit(df_log)
    print("[학습] 완료")

    print("[평가] 교차검증 중...")
    df_cv   = cross_validation(m, initial="365 days", period="30 days",
                               horizon="7 days", disable_tqdm=True)
    df_perf = performance_metrics(df_cv, rolling_window=1)

    for col in ["smape", "mape", "mdape"]:
        if col in df_perf.columns:
            val = df_perf[col].mean()
            print(f"[평가] {col.upper()} = {val:.4f}  ({'통과' if val < MAPE_THRESHOLD else '미달'})")
            return m, val

    raise ValueError("성능 지표 컬럼 없음")


# ── Pod 계산 (yhat_upper 기반) ─────────────────────────────────

def _to_pods(rps: float) -> int:
    """신뢰구간 상한(yhat_upper) 기준으로 Pod 수 결정 — 과소 프로비저닝 억제"""
    eff = CAPACITY_PER_POD * (1 - SAFETY_MARGIN)
    return max(MIN_PODS, min(math.ceil(max(rps, 0) / eff), MAX_PODS))


# ── 예측 ──────────────────────────────────────────────────────

def predict_24h(model: Prophet, df: pd.DataFrame) -> pd.DataFrame:
    last_ds      = df["ds"].max()
    future_dates = pd.date_range(start=last_ds + pd.Timedelta(hours=1), periods=24, freq="1h")
    future       = pd.DataFrame({"ds": future_dates})

    last_window             = df[df["ds"] >= last_ds - pd.Timedelta(hours=24)]
    future["is_monsoon"]    = int(last_window["is_monsoon"].mean().round())
    future["typhoon_index"] = last_window["typhoon_index"].mean()

    raw      = model.predict(future)
    forecast = inverse_transform(raw[["ds", "yhat", "yhat_lower", "yhat_upper"]])
    forecast["required_pods"] = forecast["yhat_upper"].apply(_to_pods)
    print(f"[예측] 24시간  ({future_dates[0].date()} ~ {future_dates[-1].date()})")
    return forecast


def predict_annual(model: Prophet, year: int = 2025) -> pd.DataFrame:
    dates  = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31 23:00", freq="1h")
    future = pd.DataFrame({"ds": dates})

    future["is_monsoon"] = (
        (future["ds"] >= f"{year}-06-25") & (future["ds"] <= f"{year}-07-25")
    ).astype(int)

    future["typhoon_index"] = 0.0
    rng      = np.random.default_rng(year)
    month    = int(rng.integers(8, 10))
    day      = int(rng.integers(1, 26))
    landfall = pd.Timestamp(f"{year}-{month:02d}-{day:02d}")
    for offset, idx_val in [(-1, 0.1), (0, 0.9), (1, 0.7), (2, 0.3), (3, 0.1)]:
        target = landfall + pd.Timedelta(days=offset)
        mask   = future["ds"].dt.date == target.date()
        future.loc[mask, "typhoon_index"] = np.maximum(
            future.loc[mask, "typhoon_index"].values, idx_val
        )

    raw      = model.predict(future)
    forecast = inverse_transform(raw[["ds", "yhat", "yhat_lower", "yhat_upper"]])
    forecast["required_pods"] = forecast["yhat_upper"].apply(_to_pods)

    out = DATA_DIR / f"predictions_optuna_annual_{year}.csv"
    forecast.to_csv(out, index=False)
    print(f"[연간 예측] {year}년  Pod 범위: {forecast['required_pods'].min()}~{forecast['required_pods'].max()}개")
    print(f"[저장] {out.name}")
    return forecast


# ── 저장 ──────────────────────────────────────────────────────

def save_model(model: Prophet, mape: float, cps: float, sps: float, hps: float) -> None:
    path = MODEL_DIR / f"{MODEL_NAME}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)

    runs_path = MODEL_DIR / "runs.json"
    runs      = json.loads(runs_path.read_text()) if runs_path.exists() else []
    runs.append({
        "model_name":              MODEL_NAME,
        "trained_at":              datetime.now().isoformat(),
        "mape":                    round(mape, 6),
        "stage":                   "Production" if mape < MAPE_THRESHOLD else "Staging",
        "transform":               "log1p",
        "pod_basis":               "yhat_upper",
        "changepoint_prior_scale": cps,
        "seasonality_prior_scale": sps,
        "holidays_prior_scale":    hps,
        "path":                    str(path),
    })
    runs_path.write_text(json.dumps(runs, indent=2, ensure_ascii=False))

    stage = "Production" if mape < MAPE_THRESHOLD else "Staging"
    print(f"[저장] {path.name}  stage={stage}")


# ── 메인 ──────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Optuna 모델: Prophet + log 변환 + Optuna 최적화 + yhat_upper")
    print("=" * 60)

    cps, sps, hps = load_best_params()
    df            = load_data()

    model, mape = train(df, cps, sps, hps)
    save_model(model, mape, cps, sps, hps)

    forecast_24h = predict_24h(model, df)
    out_24h      = DATA_DIR / "predictions_optuna_request_rate.csv"
    forecast_24h.to_csv(out_24h, index=False)
    print(f"[저장] {out_24h.name}  (24 rows)")

    print("\n" + "=" * 60)
    print("  연간 예측 (2025년)")
    print("=" * 60)
    predict_annual(model, year=2025)

    peak_rps = forecast_24h["yhat_upper"].max()
    headroom = (TOTAL_CLUSTER_CAPACITY - peak_rps) / TOTAL_CLUSTER_CAPACITY

    print("\n" + "=" * 60)
    print("  예측 결과 요약 (향후 24시간, yhat_upper 기준)")
    print("=" * 60)
    print(f"  예측 RPS 범위 (yhat):       {forecast_24h['yhat'].min():.1f} ~ {forecast_24h['yhat'].max():.1f} 건/분")
    print(f"  예측 RPS 범위 (yhat_upper): {forecast_24h['yhat_lower'].min():.1f} ~ {forecast_24h['yhat_upper'].max():.1f} 건/분")
    print(f"  필요 Pod 범위: {forecast_24h['required_pods'].min()} ~ {forecast_24h['required_pods'].max()} 개")
    print(f"  피크 시각:     {forecast_24h.loc[forecast_24h['yhat_upper'].idxmax(), 'ds']}")
    print(f"  Headroom:      {headroom:.1%}  ({'정상' if headroom >= SAFETY_MARGIN else '위험'})")
    print(f"  SMAPE:         {mape:.4f}  ({'통과' if mape < MAPE_THRESHOLD else '미달'})")
    print("=" * 60)


if __name__ == "__main__":
    main()
