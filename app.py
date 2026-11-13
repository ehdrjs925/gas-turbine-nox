"""가스터빈 NOx 배출 데이터 탐색 및 다중 선형회귀(Streamlit).

앱과 같은 폴더 또는 GitHub 저장소에 있는 2011~2015년 CSV를 사용합니다.
"""

from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="가스터빈 NOx 탐색 및 예측",
    page_icon="📊",
    layout="wide",
)

BASE_URL = "https://raw.githubusercontent.com/ehdrjs925/gas-turbine-nox/main"
DATA_DIR = Path(__file__).resolve().parent
TRAIN_YEARS = (2011, 2012, 2013)
TEST_YEARS = (2014, 2015)
YEARS = list(TRAIN_YEARS + TEST_YEARS)
FEATURES = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP"]
ALL_NUMERIC = FEATURES + ["CO", "NOX"]
DESCRIPTIONS = {
    "AT": "외부 기온",
    "AP": "외부 기압",
    "AH": "외부 습도",
    "AFDP": "공기 필터 차압",
    "GTEP": "가스터빈 배기압",
    "TIT": "터빈 입구 온도",
    "TAT": "터빈 출구 온도",
    "TEY": "터빈 에너지 생산량",
    "CDP": "압축기 토출압",
    "CO": "일산화탄소 배출 농도(별도 측정 결과)",
    "NOX": "질소산화물 배출 농도(예측 목표)",
}


@st.cache_data(ttl=3600, show_spinner="연도별 CSV 파일 5개를 읽고 있습니다…")
def load_data():
    """원본 자료를 유지하면서 연도와 숫자 변환 문제를 함께 확인합니다."""
    frames = []
    file_rows = []
    conversion_issues = []

    for year in YEARS:
        filename = f"gt_{year}.csv"
        local_file = DATA_DIR / filename
        if local_file.is_file():
            frame = pd.read_csv(local_file, encoding="utf-8")
        else:
            with urlopen(f"{BASE_URL}/{filename}", timeout=15) as response:
                frame = pd.read_csv(response, encoding="utf-8")
        frame.columns = frame.columns.str.strip()
        missing_columns = [col for col in ALL_NUMERIC if col not in frame.columns]
        if missing_columns:
            raise ValueError(f"gt_{year}.csv에 필요한 열이 없습니다: {', '.join(missing_columns)}")

        # 데이터가 숫자로 해석되지 않는 경우, 삭제하지 않고 결측값으로 표시해 점검합니다.
        for col in ALL_NUMERIC:
            original = frame[col]
            converted = pd.to_numeric(original, errors="coerce")
            bad = int((original.notna() & converted.isna()).sum())
            if bad:
                conversion_issues.append({"연도": year, "변수": col, "숫자로 해석 불가": bad})
            frame[col] = converted

        frame = frame[ALL_NUMERIC].copy()
        frame["year"] = year
        frames.append(frame)
        file_rows.append({"연도": year, "관측값 개수": len(frame), "원본 변수 개수": len(ALL_NUMERIC)})

    combined = pd.concat(frames, ignore_index=True)
    return combined, pd.DataFrame(file_rows), pd.DataFrame(conversion_issues)


def summary_table(df):
    summary = df[ALL_NUMERIC].agg(["count", "mean", "median", "std", "min", "max"]).T
    summary["결측값"] = df[ALL_NUMERIC].isna().sum()
    summary = summary.rename(
        columns={
            "count": "유효 관측값", "mean": "평균", "median": "중앙값",
            "std": "표준편차", "min": "최솟값", "max": "최댓값",
        }
    )
    summary.insert(0, "우리말 뜻", [DESCRIPTIONS[c] for c in summary.index])
    summary.index.name = "변수"
    return summary.reset_index().round(3)


def iqr_outlier_info(series):
    values = series.dropna()
    if values.empty:
        return None
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    count = int(((values < lower) | (values > upper)).sum())
    return {"q1": q1, "q3": q3, "iqr": iqr, "lower": lower, "upper": upper, "count": count}


st.title("가스터빈 NOx 배출 데이터 탐색 및 예측")
st.caption("데이터 출처: UCI Gas Turbine CO and NOx Emission Data Set / GitHub: ehdrjs925/gas-turbine-nox")

try:
    data, yearly_counts, conversion_issues = load_data()
except Exception as error:
    st.error("CSV를 불러오지 못했습니다. 앱과 같은 폴더의 CSV 파일명이나 GitHub 주소를 확인해 주세요.")
    st.code(str(error))
    st.stop()

st.sidebar.header("탐색 조건")
year_selection = st.sidebar.selectbox("분석에 사용할 연도", ["전체 기간"] + YEARS)
df = data if year_selection == "전체 기간" else data.loc[data["year"] == year_selection].copy()
st.sidebar.caption("연도를 바꾸면 주요 통계·분포·상관관계가 함께 바뀝니다.")

n_missing = int(df[ALL_NUMERIC].isna().sum().sum())
n_duplicates = int(df.duplicated(subset=ALL_NUMERIC + ["year"]).sum())
m1, m2, m3, m4 = st.columns(4)
m1.metric("선택된 관측값", f"{len(df):,}개")
m2.metric("원본 변수", f"{len(ALL_NUMERIC)}개")
m3.metric("결측값(셀)", f"{n_missing:,}개")
m4.metric("중복 행", f"{n_duplicates:,}개")

intro, dist, relations, raw, regression = st.tabs([
    "데이터 구조·전처리", "NOx 농도 분포", "변수 간 관계", "원본 자료·탐구 질문", "다중 선형회귀",
])

with intro:
    st.subheader("연도별 데이터 크기")
    st.dataframe(yearly_counts, hide_index=True, width="stretch")
    chart_counts = px.bar(
        yearly_counts.assign(연도=yearly_counts["연도"].astype(str)),
        x="연도", y="관측값 개수", text="관측값 개수",
        title="연도별 관측값 개수",
    )
    st.plotly_chart(chart_counts, width="stretch")

    st.subheader("변수의 의미")
    meanings = pd.DataFrame([
        {
            "변수": column, "우리말 의미": DESCRIPTIONS[column],
            "프로젝트에서의 역할": (
                "예측 목표(Y)" if column == "NOX"
                else "별도 배출 측정값(초기 입력에서 제외)" if column == "CO"
                else "독립 변수 후보(X)"
            ),
        }
        for column in ALL_NUMERIC
    ])
    st.dataframe(meanings, hide_index=True, width="stretch")
    st.caption("year는 CSV 파일명에서 부여한 연도 정보로, 원본 파일의 11개 측정 변수에는 포함되지 않습니다.")

    st.subheader("결측값과 중복값 점검")
    missing_table = pd.DataFrame({
        "변수": ALL_NUMERIC,
        "결측값 개수": [int(df[c].isna().sum()) for c in ALL_NUMERIC],
        "결측 비율(%)": [round(df[c].isna().mean() * 100, 3) for c in ALL_NUMERIC],
    })
    st.dataframe(missing_table, hide_index=True, width="stretch")
    if conversion_issues.empty:
        st.success("숫자로 해석하지 못한 데이터는 발견되지 않았습니다.")
    else:
        st.warning("숫자 변환이 불가능한 값이 발견되었습니다. 아래 목록을 확인하세요.")
        st.dataframe(conversion_issues, hide_index=True, width="stretch")
    st.caption(
        "중복 행은 같은 연도에서 11개 측정값이 모두 일치하는 행을 뜻합니다. "
        "탐색 화면은 중복 행과 결측값을 포함한 자료의 상태를 보여 줍니다."
    )

    st.subheader("기초 통계량")
    st.dataframe(summary_table(df), hide_index=True, width="stretch")
    st.caption("평균·표준편차 등은 결측값을 제외하고 계산합니다.")

with dist:
    st.subheader("NOx 배출 농도의 분포")
    valid_nox = df["NOX"].dropna()
    if valid_nox.empty:
        st.warning("선택된 연도에서 NOx 자료가 없어 분포를 그릴 수 없습니다.")
    else:
        bins = st.slider("히스토그램 구간 수", min_value=10, max_value=80, value=40, step=5)
        hist = px.histogram(
            df, x="NOX", nbins=bins, marginal="box",
            labels={"NOX": "NOx 배출 농도 (mg/m³)", "count": "관측값 개수"},
            title=f"NOx 배출 농도 히스토그램 ({year_selection})",
        )
        hist.update_layout(yaxis_title="관측값 개수", bargap=0.05)
        st.plotly_chart(hist, width="stretch")

        st.subheader("IQR 기준 이상치 탐색")
        o = iqr_outlier_info(df["NOX"])
        a, b, c = st.columns(3)
        a.metric("Q1 (1사분위수)", f"{o['q1']:.2f}")
        b.metric("Q3 (3사분위수)", f"{o['q3']:.2f}")
        c.metric("IQR 기준 이상치", f"{o['count']:,}개")
        st.write(f"참고 범위: **{o['lower']:.2f} ~ {o['upper']:.2f} mg/m³** (Q1 − 1.5×IQR ~ Q3 + 1.5×IQR)")
        st.caption("IQR 기준 밖의 값이 모두 측정 오류인 것은 아닙니다. 이상치는 제거하지 않고 원인을 검토합니다.")

    st.subheader("연도별 NOx 농도 비교")
    annual = data.groupby("year")["NOX"].agg(["count", "mean", "median", "std", "min", "max"]).round(3).reset_index()
    annual.columns = ["연도", "관측값", "평균", "중앙값", "표준편차", "최솟값", "최댓값"]
    st.dataframe(annual, hide_index=True, width="stretch")
    annual_box = px.box(
        data.assign(연도=data["year"].astype(str)), x="연도", y="NOX",
        labels={"NOX": "NOx 배출 농도 (mg/m³)"}, title="2011~2015년 NOx 농도 분포 비교", points=False,
    )
    st.plotly_chart(annual_box, width="stretch")
    st.caption("연도 간 분포 차이는 확인할 수 있지만, 시간 순서별 측정 시각은 원본 CSV에 제공되지 않습니다.")

with relations:
    st.subheader("독립 변수와 NOx의 상관관계")
    st.caption("CO는 별도 배출 측정값이므로 초기 예측 모델의 독립 변수 후보에서 제외합니다.")
    correlation = df[FEATURES + ["NOX"]].corr(method="pearson")
    heatmap = px.imshow(
        correlation, zmin=-1, zmax=1,
        color_continuous_scale="RdBu_r", text_auto=".2f", aspect="auto",
        labels={"color": "상관계수"}, title=f"피어슨 상관계수 히트맵 ({year_selection})",
    )
    heatmap.update_layout(height=610)
    st.plotly_chart(heatmap, width="stretch")
    st.caption("상관계수의 크기는 선형적 관계의 강도를 나타냅니다. 상관관계는 인과관계를 뜻하지 않습니다.")

    by_nox = correlation["NOX"].drop("NOX").reset_index()
    by_nox.columns = ["독립 변수", "NOx와의 상관계수"]
    by_nox["우리말 의미"] = by_nox["독립 변수"].map(DESCRIPTIONS)
    by_nox["절댓값"] = by_nox["NOx와의 상관계수"].abs()
    by_nox = by_nox.sort_values("절댓값", ascending=False)
    st.dataframe(by_nox.drop(columns="절댓값").round(3), hide_index=True, width="stretch")

    selected = st.selectbox("산점도로 살펴볼 독립 변수", FEATURES, format_func=lambda col: f"{col} — {DESCRIPTIONS[col]}")
    corr_value = correlation.loc[selected, "NOX"]
    if pd.isna(corr_value):
        st.warning("선택한 변수의 상관계수를 계산할 수 없습니다(자료 부족 또는 값의 변화가 없음).")
    else:
        st.metric(f"{selected}와 NOx의 상관계수", f"{corr_value:.3f}")

    # 시각화에만 표본을 사용합니다. 모든 통계와 상관계수는 전체 선택 자료에서 계산합니다.
    scatter = df[[selected, "NOX", "year"]].dropna()
    if not scatter.empty:
        if len(scatter) > 4500:
            scatter = scatter.sample(n=4500, random_state=42)
        scatter = scatter.assign(연도=scatter["year"].astype(str))
        scatter_fig = px.scatter(
            scatter, x=selected, y="NOX", color="연도", opacity=0.5,
            title=f"{DESCRIPTIONS[selected]}와 NOx 배출 농도의 관계",
            labels={selected: f"{DESCRIPTIONS[selected]} ({selected})", "NOX": "NOx 배출 농도 (mg/m³)"},
        )
        st.plotly_chart(scatter_fig, width="stretch")
        st.caption("산점도는 화면 속도를 위해 최대 4,500개만 표시합니다. 상관계수는 선택된 연도의 전체 유효 관측값으로 계산합니다.")
    else:
        st.info("산점도를 표시할 유효한 자료가 없습니다.")
    st.info(
        "해석할 때 확인할 점: 산점도가 직선에 가깝게 나타나는가? "
        "독립 변수끼리 상관관계가 높은가? 연도별로 데이터 분포가 다른가?"
    )

with raw:
    st.subheader("실제 데이터 미리보기")
    st.dataframe(df.head(30), hide_index=True, width="stretch")
    st.caption("모든 열의 자료가 숫자인지, 연도별로 정상 결합되었는지 확인해 보세요.")

    st.subheader("직접 생각해 볼 탐구 질문")
    st.markdown(
        """
1. NOx 농도는 어떤 범위에 가장 많이 분포하는가?
2. NOx와 상관관계가 큰 운전 조건은 무엇인가? 산점도도 이를 뒷받침하는가?
3. 연도별 NOx 농도 분포가 달라 보이는가? 근거는 무엇인가?
4. 이상치는 실제 드문 운전 상태일 가능성이 있는가? 삭제 전에 무엇을 확인해야 하는가?
5. 실제값과 예측값 사이의 차이가 어느 농도 구간에서 크게 나타나는가?
"""
    )
    st.warning(
        "이 질문의 답과 수행평가 보고서·일지는 수업 시간에 직접 작성하세요."
    )


with regression:
    st.subheader("운전 조건 9개로 NOx 농도 예측")
    st.caption("입력 변수 " + ", ".join(FEATURES) + " · 예측 대상 NOX")
    st.caption("사이드바의 연도 선택은 탐색용입니다. 모델의 학습·평가 기간은 아래와 같습니다.")

    model_columns = FEATURES + ["NOX"]
    train = data.loc[data["year"].isin(TRAIN_YEARS), model_columns]
    test = data.loc[data["year"].isin(TEST_YEARS), ["year"] + model_columns]

    # 각 기간에서 모델에 필요한 값이 모두 유효한 관측 행을 사용합니다.
    train_ready = train.replace([float("inf"), float("-inf")], float("nan")).dropna()
    test_ready = test.replace([float("inf"), float("-inf")], float("nan")).dropna()

    st.dataframe(
        pd.DataFrame({
            "용도": ["학습", "평가"],
            "기간": ["2011~2013년", "2014~2015년"],
            "사용한 행": [len(train_ready), len(test_ready)],
            "결측값·무한값으로 제외한 행": [len(train) - len(train_ready), len(test) - len(test_ready)],
        }),
        hide_index=True,
        width="stretch",
    )

    if train_ready.empty or test_ready.empty:
        st.warning("유효한 학습 자료와 평가 자료가 모두 있어야 모델을 평가할 수 있습니다.")
    else:
        model = LinearRegression()
        model.fit(train_ready[FEATURES], train_ready["NOX"])
        actual = test_ready["NOX"]
        predicted = model.predict(test_ready[FEATURES])

        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        r2 = r2_score(actual, predicted) if actual.nunique() >= 2 else None
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE (mg/m³)", f"{mae:.3f}", help="실제값과 예측값 차이의 절댓값을 평균한 값입니다.")
        c2.metric("MSE ((mg/m³)²)", f"{mse:.3f}", help="실제값과 예측값 차이를 제곱한 뒤 평균한 값입니다.")
        c3.metric(
            "R²",
            f"{r2:.3f}" if r2 is not None else "계산 불가",
            help="1에 가까울수록 평가 자료에 잘 맞습니다. 실제값에 변화가 없으면 정의되지 않습니다.",
        )

        points = pd.DataFrame({
            "실제 NOx 농도 (mg/m³)": actual.to_numpy(),
            "예측 NOx 농도 (mg/m³)": predicted,
            "평가 연도": test_ready["year"].astype(str).to_numpy(),
        })
        fig = px.scatter(
            points,
            x="실제 NOx 농도 (mg/m³)",
            y="예측 NOx 농도 (mg/m³)",
            color="평가 연도",
            opacity=0.45,
            render_mode="webgl",
            title="실제값과 예측값 비교",
        )
        lower = min(float(actual.min()), float(predicted.min()))
        upper = max(float(actual.max()), float(predicted.max()))
        padding = max((upper - lower) * 0.04, 0.1)
        axis_range = [lower - padding, upper + padding]
        fig.add_shape(
            type="line", x0=axis_range[0], x1=axis_range[1],
            y0=axis_range[0], y1=axis_range[1],
            line={"color": "#777777", "dash": "dash"},
            layer="below",
        )
        fig.update_xaxes(range=axis_range)
        fig.update_yaxes(range=axis_range, scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig, width="stretch")
        st.caption("점선은 실제값과 예측값이 같은 위치입니다. 점선 아래는 과소 예측, 위는 과대 예측입니다.")
