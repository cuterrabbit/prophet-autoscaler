"""
모듈 3: 재학습 파이프라인
참조: module-spec/module-spec-3.md

실행: python3 src/retrain.py
전제: data/dummy_request_rate.csv     (기존 학습 데이터)
      data/dummy_request_rate_new.csv  (추가 데이터, generate_dummy_data.py --start ... 로 생성)

출력: models/request-rate-forecast-retrained.pkl  (재학습 모델)
      data/predictions_retrained.csv              (재학습 후 24시간 예측)
      models/runs.json                            (실험 기록 업데이트)
"""
import json
import math
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

CAPACITY_PER_POD       = 80.0
SAFETY_MARGIN          = 0.2
TOTAL_CLUSTER_CAPACITY = 500.0
MAPE_THRESHOLD         = 0.15


# ── 데이터 ────────────────────────────────────────────────────

def merge_data() -> pd.DataFrame:
    """기존 데이터 + 추가 데이터 병합"""
    orig_path = DATA_DIR / "dummy_request_rate.csv"
    new_path  = DATA_DIR / "dummy_request_rate_new.csv"

    if not new_path.exists():
        raise FileNotFoundError(
            f"{new_path.name} 없음.\n"
            "먼저 실행: python3 src/generate_dummy_data.py --start YYYY-MM-DD --end YYYY-MM-DD"
        )

    orig = pd.read_csv(orig_path, parse_dates=["ds"])
    new  = pd.read_csv(new_path,  parse_dates=["ds"])

    merged = (
        pd.concat([orig, new], ignore_index=True)
        .drop_duplicates(subset="ds")
        .sort_values("ds")
        .reset_index(drop=True)
    )

    print(f"[병합] 기존: {len(orig):,}행  추가: {len(new):,}행  → 합계: {len(merged):,}행")
    print(f"       기간: {merged['ds'].min().date()} ~ {merged['ds'].max().date()}")

    merged_path = DATA_DIR / "dummy_request_rate_merged.csv"
    merged.to_csv(merged_path, index=False)
    print(f"[병합] 저장: {merged_path.name}")
    return merged


# ── 모델 ──────────────────────────────────────────────────────

def build_model() -> Prophet:
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
    )
    m.add_country_holidays(country_name="KR")
    m.add_regressor("is_monsoon")
    m.add_regressor("typhoon_index")
    return m


def retrain(df: pd.DataFrame) -> tuple[Prophet, float]:
    print("\n[재학습] 모델 학습 시작...")
    m = build_model()
    m.fit(df)
    print("[재학습] 완료")

    print("[평가] 교차검증 중...")
    df_cv = cross_validation(
        m, initial="365 days", period="30 days", horizon="7 days", disable_tqdm=True
    )
    df_perf = performance_metrics(df_cv, rolling_window=1)

    for col in ["mape", "smape", "mdape"]:
        if col in df_perf.columns:
            mape = df_perf[col].mean()
            print(f"[평가] {col.upper()} = {mape:.4f} ({'통과' if mape < MAPE_THRESHOLD else '미달'})")
            return m, mape

    raise ValueError("성능 지표 컬럼 없음")


def predict(model: Prophet, df: pd.DataFrame, horizon_hours: int = 24) -> pd.DataFrame:
    last_ds = df["ds"].max()
    future_dates = pd.date_range(
        start=last_ds + pd.Timedelta(hours=1),
        periods=horizon_hours,
        freq="1h",
    )
    future = pd.DataFrame({"ds": future_dates})

    last_window = df[df["ds"] >= last_ds - pd.Timedelta(hours=24)]
    future["is_monsoon"]    = int(last_window["is_monsoon"].mean().round())
    future["typhoon_index"] = last_window["typhoon_index"].mean()

    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def load_old_predictions() -> pd.DataFrame:
    path = DATA_DIR / "predictions_request_rate.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["ds"])


def calculate_required_pods(predicted_rps: float) -> int:
    effective = CAPACITY_PER_POD * (1 - SAFETY_MARGIN)
    return math.ceil(max(predicted_rps, 0) / effective)


def save_runs(mape: float, stage: str) -> None:
    runs_path = MODEL_DIR / "runs.json"
    runs = json.loads(runs_path.read_text()) if runs_path.exists() else []
    runs.append({
        "model_name": "request-rate-forecast-retrained",
        "trained_at": datetime.now().isoformat(),
        "mape": round(mape, 6),
        "stage": stage,
        "path": str(MODEL_DIR / "request-rate-forecast-retrained.pkl"),
    })
    runs_path.write_text(json.dumps(runs, indent=2, ensure_ascii=False))


# ── 메인 ──────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  모듈 3: 재학습 파이프라인 (request_rate)")
    print("=" * 55)

    # 1. 데이터 병합
    df = merge_data()

    # 2. 재학습
    model, mape = retrain(df)

    # 3. 모델 저장
    stage = "Production" if mape < MAPE_THRESHOLD else "Staging"
    model_path = MODEL_DIR / "request-rate-forecast-retrained.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    save_runs(mape, stage)
    print(f"[저장] {model_path.name}  stage={stage}")

    # 4. 재학습 후 예측
    new_forecast = predict(model, df)
    new_forecast["required_pods"] = new_forecast["yhat"].apply(calculate_required_pods)
    new_forecast.to_csv(DATA_DIR / "predictions_retrained.csv", index=False)
    print(f"[저장] predictions_retrained.csv")

    # 5. 이전 예측과 비교
    old_forecast = load_old_predictions()

    print("\n" + "=" * 55)
    print("  재학습 전후 비교 (향후 24시간 예측)")
    print("=" * 55)

    if not old_forecast.empty and "required_pods" in old_forecast.columns:
        old_pods = old_forecast["required_pods"]
        new_pods = new_forecast["required_pods"]
        print(f"  {'구분':<20} {'재학습 전':>12} {'재학습 후':>12}")
        print(f"  {'-'*44}")
        print(f"  {'예측 RPS 평균 (건/분)':<20} {old_forecast['yhat'].mean():>12.1f} {new_forecast['yhat'].mean():>12.1f}")
        print(f"  {'예측 RPS 최대 (건/분)':<20} {old_forecast['yhat'].max():>12.1f} {new_forecast['yhat'].max():>12.1f}")
        print(f"  {'필요 Pod 최소':<20} {old_pods.min():>12} {new_pods.min():>12}")
        print(f"  {'필요 Pod 최대':<20} {old_pods.max():>12} {new_pods.max():>12}")
    else:
        print("  (이전 예측 없음 — 재학습 후 결과만 표시)")
        print(f"  예측 RPS: {new_forecast['yhat'].min():.1f} ~ {new_forecast['yhat'].max():.1f} 건/분")
        print(f"  필요 Pod: {new_forecast['required_pods'].min()} ~ {new_forecast['required_pods'].max()} 개")

    print(f"\n  모델 성능: {mape:.4f} ({'✅ Production 승격' if stage == 'Production' else '⚠️  Staging 유지'})")
    print("=" * 55)
    print("\n시각화: python3 src/compare_plot.py")


if __name__ == "__main__":
    main()
