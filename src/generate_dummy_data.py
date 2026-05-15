"""
모듈 1: 더미 데이터 생성
참조: module-spec/module-spec-1-2.md

실행: python src/generate_dummy_data.py
출력: data/dummy_request_rate.csv
      data/dummy_cpu_utilization.csv
      data/dummy_anomaly_events.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

# ── 기본값 ────────────────────────────────────────────────────
BASE_REQUEST_RATE = 100.0  # 건/분 (시간 평균 기준)
BASE_CPU = 50.0            # % (기준 CPU 사용률)

# 월별 계절 가중치 (작물별 파종/상환 패턴 합산)
MONTHLY_WEIGHT = {
    1: 0.20, 2: 0.30, 3: 1.00, 4: 0.85,
    5: 0.60, 6: 0.50, 7: 0.35, 8: 0.45,
    9: 0.45, 10: 0.35, 11: 0.35, 12: 0.20,
}

# 요일별 가중치 (0=월 ~ 6=일)
WEEKLY_WEIGHT = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.7, 6: 0.4}


def _hourly_weight(hour: int) -> float:
    if 6 <= hour < 9:   return 0.90  # 농작업 시작 전 주문 확인
    if 9 <= hour < 12:  return 0.60  # 오전 구매
    if 12 <= hour < 14: return 0.30  # 점심
    if 14 <= hour < 18: return 0.55  # 오후 구매
    if 18 <= hour < 21: return 0.85  # 저녁 구매/상환 조회 피크
    return 0.10                       # 야간


# ── 함수 ──────────────────────────────────────────────────────

def generate_base_pattern(start_date: str, end_date: str, freq: str = "1h") -> pd.DataFrame:
    """기본 시계열 뼈대 생성 (타임스탬프 인덱스)"""
    timestamps = pd.date_range(start=start_date, end=end_date, freq=freq)
    return pd.DataFrame({
        "ds": timestamps,
        "is_monsoon": 0,
        "typhoon_index": 0.0,
    })


def apply_crop_seasonal_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """작물별 계절 패턴 적용 (파종/상환 피크 포함)"""
    df = df.copy()
    df["_seasonal"] = df["ds"].dt.month.map(MONTHLY_WEIGHT)
    return df


def apply_weekly_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """요일별 패턴 적용 (월~금 높음, 주말 낮음)"""
    df = df.copy()
    df["_weekly"] = df["ds"].dt.dayofweek.map(WEEKLY_WEIGHT)
    return df


def apply_daily_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """시간대별 패턴 적용 (오전 6~9시, 저녁 18~21시 피크)"""
    df = df.copy()
    df["_daily"] = df["ds"].dt.hour.apply(_hourly_weight)
    return df


def apply_weather_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """
    기상 패턴 적용 + regressor 컬럼 생성
    - is_monsoon: 6/25 ~ 7/25 (0/1)
    - typhoon_index: 8~9월 태풍 영향 지수 (0.0~1.0)
    """
    df = df.copy()
    rng = np.random.default_rng(42)

    for year in sorted(df["ds"].dt.year.unique()):
        # 장마
        mask_monsoon = (df["ds"] >= f"{year}-06-25") & (df["ds"] <= f"{year}-07-25")
        df.loc[mask_monsoon, "is_monsoon"] = 1

        # 태풍: 연 1~2회, 8~9월 중 랜덤 상륙
        n_typhoons = rng.integers(1, 3)
        for _ in range(n_typhoons):
            month = rng.integers(8, 10)
            day = rng.integers(1, 26)
            landfall = pd.Timestamp(f"{year}-{month:02d}-{day:02d}")

            # 상륙 전날(-1), 당일(0), 이후(1~3) 영향 지수
            for offset, idx_val in [(-1, 0.1), (0, 0.9), (1, 0.7), (2, 0.3), (3, 0.1)]:
                target = landfall + pd.Timedelta(days=offset)
                mask_t = df["ds"].dt.date == target.date()
                df.loc[mask_t, "typhoon_index"] = np.maximum(
                    df.loc[mask_t, "typhoon_index"].values, idx_val
                )

    return df


def inject_anomalies(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    RCA 테스트용 이상치 구간 삽입
    - 시나리오 1: 파종기 트래픽 급증 (3월, 매년)
    - 시나리오 2: 태풍 직후 긴급 구매 급증 (9월, 매년)
    - 시나리오 3: 장마철 불규칙 (7월, 첫해)
    """
    df = df.copy()
    df["_anomaly_boost"] = 1.0
    events = []
    rng = np.random.default_rng(99)

    for year in sorted(df["ds"].dt.year.unique()):
        # 시나리오 1: 파종기 급증 (3월 10~20일, +40%)
        s1 = pd.Timestamp(f"{year}-03-10")
        e1 = pd.Timestamp(f"{year}-03-20 23:00")
        mask1 = (df["ds"] >= s1) & (df["ds"] <= e1)
        df.loc[mask1, "_anomaly_boost"] *= 1.4
        events.append({
            "scenario": 1, "year": year,
            "start": str(s1), "end": str(e1),
            "description": "파종기 트래픽 급증 (+40%)",
        })

        # 시나리오 2: 태풍 직후 급증 (9월 2~4일, +50%)
        s2 = pd.Timestamp(f"{year}-09-02")
        e2 = pd.Timestamp(f"{year}-09-04 23:00")
        mask2 = (df["ds"] >= s2) & (df["ds"] <= e2)
        df.loc[mask2, "_anomaly_boost"] *= 1.5
        events.append({
            "scenario": 2, "year": year,
            "start": str(s2), "end": str(e2),
            "description": "태풍 직후 긴급 구매 급증 (+50%)",
        })

    # 시나리오 3: 장마철 불규칙 (첫해 7월 5~15일, ±50% 랜덤)
    first_year = sorted(df["ds"].dt.year.unique())[0]
    s3 = pd.Timestamp(f"{first_year}-07-05")
    e3 = pd.Timestamp(f"{first_year}-07-15 23:00")
    mask3 = (df["ds"] >= s3) & (df["ds"] <= e3)
    df.loc[mask3, "_anomaly_boost"] *= rng.uniform(0.5, 1.5, mask3.sum())
    events.append({
        "scenario": 3, "year": first_year,
        "start": str(s3), "end": str(e3),
        "description": "장마철 불규칙 트래픽 (×0.5~1.5 랜덤)",
    })

    return df, pd.DataFrame(events)


def _build_y(df: pd.DataFrame, base: float, rng_seed: int, clip_max: float) -> np.ndarray:
    """패턴 조합 → 최종 y 값 생성"""
    rng = np.random.default_rng(rng_seed)
    noise = rng.normal(0, base * 0.03, len(df))

    # 태풍 당일 급감(0.4), 직후 급증(1.3)
    typhoon_effect = np.where(
        df["typhoon_index"] >= 0.8, 0.4,
        np.where(df["typhoon_index"] >= 0.3, 1.3, 1.0)
    )

    # 장마 기간 ±20% 불규칙
    monsoon_rng = np.random.default_rng(rng_seed + 1)
    monsoon_effect = np.where(
        df["is_monsoon"].values == 1,
        monsoon_rng.uniform(0.8, 1.2, len(df)),
        1.0,
    )

    y = (
        base
        * df["_seasonal"].values
        * df["_weekly"].values
        * df["_daily"].values
        * typhoon_effect
        * monsoon_effect
        * df["_anomaly_boost"].values
        + noise
    )
    return np.clip(y, 0, clip_max)


def save_to_csv(df: pd.DataFrame, path: Path) -> None:
    """Prophet 형식으로 CSV 저장"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  저장: {path.name}  ({len(df):,} rows)")


# ── 메인 ──────────────────────────────────────────────────────

def main():
    START = "2023-01-01"
    END   = "2024-12-31"
    OUT   = Path(__file__).resolve().parent.parent / "data"

    print("=== 더미 데이터 생성 시작 ===")
    print(f"기간: {START} ~ {END}\n")

    # 1. 패턴 조합
    df = generate_base_pattern(START, END)
    df = apply_crop_seasonal_pattern(df)
    df = apply_weekly_pattern(df)
    df = apply_daily_pattern(df)
    df = apply_weather_pattern(df)
    df, anomaly_df = inject_anomalies(df)

    # 2. 요청수 (건/분)
    rr = df[["ds", "is_monsoon", "typhoon_index"]].copy()
    rr["y"] = _build_y(df, BASE_REQUEST_RATE, rng_seed=42, clip_max=500.0)
    save_to_csv(rr, OUT / "dummy_request_rate.csv")

    # 3. CPU 사용률 (%)
    cpu = df[["ds", "is_monsoon", "typhoon_index"]].copy()
    cpu["y"] = _build_y(df, BASE_CPU, rng_seed=123, clip_max=100.0)
    save_to_csv(cpu, OUT / "dummy_cpu_utilization.csv")

    # 4. 이상치 구간 정보
    save_to_csv(anomaly_df, OUT / "dummy_anomaly_events.csv")

    # 5. 요약
    print(f"\n총 {len(df):,} 시간 레코드 (2년치)")
    print(f"  request_rate  범위: {rr['y'].min():.1f} ~ {rr['y'].max():.1f} 건/분")
    print(f"  cpu_util      범위: {cpu['y'].min():.1f} ~ {cpu['y'].max():.1f} %")
    print(f"  이상치 시나리오: {len(anomaly_df)}건")
    print(f"\n저장 위치: {OUT}")
    print("=== 완료 ===")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="더미 데이터 생성")
    parser.add_argument("--start", default="2023-01-01", help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end",   default="2024-12-31", help="종료일 (YYYY-MM-DD)")
    args = parser.parse_args()

    OUT = Path(__file__).resolve().parent.parent / "data"

    # 추가 생성 모드: 기본 범위가 아니면 _new 접미사로 저장
    is_additional = (args.start != "2023-01-01" or args.end != "2024-12-31")
    suffix = "_new" if is_additional else ""

    print("=== 더미 데이터 생성 시작 ===")
    print(f"기간: {args.start} ~ {args.end}{'  (추가 데이터)' if is_additional else ''}\n")

    df = generate_base_pattern(args.start, args.end)
    df = apply_crop_seasonal_pattern(df)
    df = apply_weekly_pattern(df)
    df = apply_daily_pattern(df)
    df = apply_weather_pattern(df)
    df, anomaly_df = inject_anomalies(df)

    rr = df[["ds", "is_monsoon", "typhoon_index"]].copy()
    rr["y"] = _build_y(df, BASE_REQUEST_RATE, rng_seed=42, clip_max=500.0)
    save_to_csv(rr, OUT / f"dummy_request_rate{suffix}.csv")

    cpu = df[["ds", "is_monsoon", "typhoon_index"]].copy()
    cpu["y"] = _build_y(df, BASE_CPU, rng_seed=123, clip_max=100.0)
    save_to_csv(cpu, OUT / f"dummy_cpu_utilization{suffix}.csv")

    save_to_csv(anomaly_df, OUT / f"dummy_anomaly_events{suffix}.csv")

    print(f"\n총 {len(df):,} 시간 레코드")
    print(f"  request_rate  범위: {rr['y'].min():.1f} ~ {rr['y'].max():.1f} 건/분")
    print(f"  저장 위치: {OUT}")
    print("=== 완료 ===")
