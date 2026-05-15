"""
하이퍼파라미터 탐색 (로그 변환 + 그리드 서치 + 조합 단위 병렬화)

탐색 대상:
  - changepoint_prior_scale  : 트렌드 유연성 (클수록 피크 추종, 작을수록 안정)
  - seasonality_prior_scale  : 계절성 강도  (클수록 계절 패턴 강조)

병렬화 전략:
  - 20개 조합을 ProcessPoolExecutor로 동시 실행
  - 조합 단위 병렬이 cutpoint 단위 병렬보다 Windows에서 유리
    (spawn 오버헤드를 1회로 줄임)

실행: python src_tuned/hyperparameter_search.py
출력: data/hp_search_results.csv   — 조합별 SMAPE
      (콘솔) 최적 파라미터 출력
"""
import itertools
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# ── 탐색 그리드 ────────────────────────────────────────────────
CHANGEPOINT_SCALES = [0.01, 0.05, 0.1, 0.2, 0.3]
SEASONALITY_SCALES = [5.0, 10.0, 15.0, 20.0]
GRID = list(itertools.product(CHANGEPOINT_SCALES, SEASONALITY_SCALES))

N_WORKERS = max(1, os.cpu_count() - 1)   # 코어 수 - 1 (OS 여유분 확보)


# ── 공통 ──────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "dummy_request_rate.csv", parse_dates=["ds"])
    print(f"[로드] {len(df):,}행  ({df['ds'].min().date()} ~ {df['ds'].max().date()})")
    return df


def log_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["y"] = np.log1p(df["y"])
    return df


def build_model(cps: float, sps: float) -> Prophet:
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode="additive",
        changepoint_prior_scale=cps,
        seasonality_prior_scale=sps,
    )
    m.add_country_holidays(country_name="KR")
    m.add_regressor("is_monsoon")
    m.add_regressor("typhoon_index")
    return m


# 모듈 최상위 함수여야 ProcessPoolExecutor에서 pickle 가능
def evaluate_combo(args: tuple) -> dict:
    """단일 조합 평가 — 워커 프로세스에서 실행"""
    cps, sps, df_log = args
    warnings.filterwarnings("ignore")

    m = build_model(cps, sps)
    m.fit(df_log)
    df_cv   = cross_validation(m, initial="365 days", period="30 days", horizon="7 days", disable_tqdm=True)
    df_perf = performance_metrics(df_cv, rolling_window=1)

    smape = float("inf")
    for col in ["smape", "mape", "mdape"]:
        if col in df_perf.columns:
            smape = float(df_perf[col].mean())
            break

    return {"changepoint_prior_scale": cps, "seasonality_prior_scale": sps, "smape": smape}


# ── 병렬 그리드 서치 ──────────────────────────────────────────

def search(df: pd.DataFrame) -> pd.DataFrame:
    df_log = log_transform(df)
    total  = len(GRID)
    args   = [(cps, sps, df_log) for cps, sps in GRID]

    print(f"\n[탐색] {total}개 조합  병렬 워커: {N_WORKERS}개")
    print(f"       예상 시간: 순차 대비 약 {N_WORKERS}배 단축")
    print(f"  {'cps':>6}  {'sps':>6}  {'SMAPE':>8}  {'상태':>6}")
    print(f"  {'-'*36}")

    rows      = []
    completed = 0

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(evaluate_combo, a): a for a in args}
        for fut in as_completed(futures):
            result     = fut.result()
            completed += 1
            rows.append(result)
            best_so_far = min(r["smape"] for r in rows)
            mark = " ★" if result["smape"] == best_so_far else ""
            print(f"  {result['changepoint_prior_scale']:>6.3f}  "
                  f"{result['seasonality_prior_scale']:>6.1f}  "
                  f"{result['smape']:>8.4f}  "
                  f"{completed:>3}/{total}{mark}")

    results = pd.DataFrame(rows).sort_values("smape").reset_index(drop=True)
    out = DATA_DIR / "hp_search_results.csv"
    results.to_csv(out, index=False)
    print(f"\n[저장] {out.name}")
    return results


# ── 메인 ──────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  하이퍼파라미터 그리드 서치  (log 변환 + 조합 병렬)")
    print("=" * 55)

    df      = load_data()
    results = search(df)
    best    = results.iloc[0]

    print("\n" + "=" * 55)
    print("  탐색 결과 (상위 5개)")
    print("=" * 55)
    print(results.head(5).to_string(index=False))
    print()
    print(f"  ★ 최적 파라미터")
    print(f"    changepoint_prior_scale = {best['changepoint_prior_scale']}")
    print(f"    seasonality_prior_scale = {best['seasonality_prior_scale']}")
    print(f"    SMAPE                   = {best['smape']:.4f}")
    print()
    print("  → src_tuned/train_predict.py 에 이 값이 자동 반영됩니다.")


# Windows multiprocessing은 반드시 이 guard가 필요
if __name__ == "__main__":
    main()
