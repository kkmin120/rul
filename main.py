
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import silhouette_score

st.set_page_config(
    page_title="공중화장실 사각지대 분석",
    page_icon="🚻",
    layout="wide"
)

st.title("🚻 공중화장실 사각지대 분석")
st.caption("지역별 화장실 분포와 접근성 격차 분석")
st.markdown("""
전국 공중화장실 데이터와 주민등록인구, 국토면적 데이터를 결합하여
지역별 공중화장실 분포와 상대적인 시설 부족 지역을 분석합니다.
""")

st.divider()

RESTROOM_URL = "https://file.localdata.go.kr/file/download/public_restroom_info/info"

@st.cache_data(ttl=3600)
def load_restroom():
    response = requests.get(RESTROOM_URL, timeout=60)
    response.raise_for_status()
    return pd.read_csv(
        BytesIO(response.content),
        encoding="utf-8-sig",
        low_memory=False
    )

def read_csv_file(file):
    for enc in ["utf-8-sig", "cp949", "euc-kr"]:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except Exception:
            continue
    raise ValueError("CSV 파일을 읽을 수 없습니다.")

def clean_region(value):
    if pd.isna(value):
        return None

    value = str(value).strip()
    value = value.replace("특별자치도", "특별자치도")

    provinces = [
        "서울특별시", "부산광역시", "대구광역시",
        "인천광역시", "광주광역시", "대전광역시",
        "울산광역시", "세종특별자치시",
        "경기도", "강원특별자치도", "충청북도",
        "충청남도", "전북특별자치도", "전라남도",
        "경상북도", "경상남도", "제주특별자치도"
    ]

    short = {
        "서울": "서울특별시",
        "부산": "부산광역시",
        "대구": "대구광역시",
        "인천": "인천광역시",
        "광주": "광주광역시",
        "대전": "대전광역시",
        "울산": "울산광역시",
        "세종": "세종특별자치시",
        "강원도": "강원특별자치도",
        "전라북도": "전북특별자치도",
        "제주도": "제주특별자치도"
    }

    for old, new in short.items():
        if value.startswith(old):
            value = new + value[len(old):]

    province = None
    for p in provinces:
        if value.startswith(p):
            province = p
            break

    if province is None:
        return None

    rest = value[len(province):].strip()
    parts = rest.split()

    if province == "세종특별자치시":
        return province

    if not parts:
        return None

    # 시·군·구 단위로 통일
    district = parts[0]

    if not any(district.endswith(x) for x in ["시", "군", "구"]):
        return None

    return province + " " + district


def guess_column(df, keywords):
    for col in df.columns:
        name = str(col).replace(" ", "")
        if any(k in name for k in keywords):
            return col
    return df.columns[0]


def number_column(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("명", "", regex=False)
        .str.replace("㎢", "", regex=False)
        .str.strip(),
        errors="coerce"
    )


def make_region_data(restroom, population, area,
                     address_col, pop_region, pop_value,
                     area_region, area_value):

    restroom = restroom.copy()
    population = population.copy()
    area = area.copy()

    restroom["지역"] = restroom[address_col].apply(clean_region)
    restroom = restroom.dropna(subset=["지역"])

    toilet_count = (
        restroom.groupby("지역")
        .size()
        .reset_index(name="공중화장실수")
    )

    population["지역"] = population[pop_region].apply(clean_region)
    population["인구수"] = number_column(population[pop_value])

    pop_data = (
        population.dropna(subset=["지역", "인구수"])
        .groupby("지역")["인구수"]
        .sum()
        .reset_index()
    )

    area["지역"] = area[area_region].apply(clean_region)
    area["면적"] = number_column(area[area_value])

    area_data = (
        area.dropna(subset=["지역", "면적"])
        .groupby("지역")["면적"]
        .sum()
        .reset_index()
    )

    data = toilet_count.merge(pop_data, on="지역", how="inner")
    data = data.merge(area_data, on="지역", how="inner")

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()

    data = data[
        (data["인구수"] > 0) &
        (data["면적"] > 0)
    ].copy()

    data["인구 1만 명당 화장실"] = (
        data["공중화장실수"] / data["인구수"] * 10000
    )

    data["면적 10㎢당 화장실"] = (
        data["공중화장실수"] / data["면적"] * 10
    )

    return data


# 데이터 불러오기
st.header("1. 데이터 준비")

with st.expander("사용 데이터 출처", expanded=False):
    st.markdown("""
    - 전국 공중화장실 표준데이터: 행정안전부
    - 주민등록인구: KOSIS, 2025년
    - 행정구역별 국토면적: 2025년 자료

    공중화장실 자료는 자동으로 불러오며,
    인구와 면적 자료는 CSV 파일을 사용합니다.
    """)

try:
    with st.spinner("전국 공중화장실 데이터를 불러오는 중입니다."):
        restroom = load_restroom()
    st.success(f"공중화장실 데이터 불러오기 완료: {len(restroom):,}행")
except Exception as e:
    st.error("공중화장실 데이터를 자동으로 불러오지 못했습니다.")
    st.write("아래에서 공중화장실 CSV 파일을 직접 올려 주세요.")
    restroom_file = st.file_uploader(
        "전국 공중화장실 CSV",
        type=["csv"]
    )
    if restroom_file is None:
        st.stop()
    restroom = read_csv_file(restroom_file)

col1, col2 = st.columns(2)

with col1:
    population_file = st.file_uploader(
        "KOSIS 주민등록인구 CSV",
        type=["csv"]
    )

with col2:
    area_file = st.file_uploader(
        "행정구역별 국토면적 CSV",
        type=["csv"]
    )

if population_file is None and __import__("os").path.exists("population.csv"):
    population = pd.read_csv("population.csv", encoding="utf-8-sig")
elif population_file is not None:
    population = read_csv_file(population_file)
else:
    st.info("population.csv를 GitHub에 올리거나 위에서 파일을 선택해 주세요.")
    st.stop()

if area_file is None and __import__("os").path.exists("area.csv"):
    area = pd.read_csv("area.csv", encoding="utf-8-sig")
elif area_file is not None:
    area = read_csv_file(area_file)
else:
    st.info("area.csv를 GitHub에 올리거나 위에서 파일을 선택해 주세요.")
    st.stop()

st.success("인구 및 면적 데이터 불러오기 완료")

# 열 선택
st.header("2. 데이터 열 설정")

r_address = guess_column(
    restroom,
    ["소재지도로명주소", "소재지지번주소", "주소"]
)

p_region = guess_column(
    population,
    ["행정구역", "행정구역별", "시군구"]
)

p_value = guess_column(
    population,
    ["총인구수", "인구수", "계"]
)

a_region = guess_column(
    area,
    ["행정구역", "시군구", "지역"]
)

a_value = guess_column(
    area,
    ["국토면적", "면적", "면적(㎢)", "면적(km2)"]
)

with st.expander("열 이름 확인 및 수정", expanded=True):
    c1, c2 = st.columns(2)

    with c1:
        address_col = st.selectbox(
            "화장실 주소 열",
            restroom.columns,
            index=list(restroom.columns).index(r_address)
        )
        pop_region = st.selectbox(
            "인구 지역 열",
            population.columns,
            index=list(population.columns).index(p_region)
        )
        pop_value = st.selectbox(
            "인구수 열",
            population.columns,
            index=list(population.columns).index(p_value)
        )

    with c2:
        area_region = st.selectbox(
            "면적 지역 열",
            area.columns,
            index=list(area.columns).index(a_region)
        )
        area_value = st.selectbox(
            "국토면적 열",
            area.columns,
            index=list(area.columns).index(a_value)
        )

try:
    data = make_region_data(
        restroom, population, area,
        address_col, pop_region, pop_value,
        area_region, area_value
    )
except Exception as e:
    st.error(f"데이터 결합 오류: {e}")
    st.stop()

if len(data) < 5:
    st.error("결합된 지역 데이터가 너무 적습니다. 지역명과 열 설정을 확인해 주세요.")
    st.stop()

st.success(f"분석 가능한 지역: {len(data)}개")

# 주요 지표
st.divider()
st.header("3. 전국 공중화장실 현황")

c1, c2, c3, c4 = st.columns(4)

c1.metric("분석 지역 수", f"{len(data):,}개")
c2.metric("공중화장실 수", f"{int(data['공중화장실수'].sum()):,}개")
c3.metric("총인구", f"{int(data['인구수'].sum()):,}명")
c4.metric("평균 화장실 수", f"{data['공중화장실수'].mean():.1f}개")

st.subheader("지역별 공중화장실 개수")

fig = px.bar(
    data.sort_values("공중화장실수", ascending=False),
    x="지역",
    y="공중화장실수",
    color="공중화장실수",
    title="지역별 공중화장실 수 비교",
    labels={"지역": "지역", "공중화장실수": "화장실 수"}
)
fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig, use_container_width=True)

# 인구 및 면적 대비 분석
st.divider()
st.header("4. 인구와 면적을 고려한 접근성 분석")

c1, c2 = st.columns(2)

with c1:
    fig = px.bar(
        data.sort_values("인구 1만 명당 화장실"),
        x="지역",
        y="인구 1만 명당 화장실",
        title="인구 1만 명당 공중화장실 수",
        color="인구 1만 명당 화장실"
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = px.bar(
        data.sort_values("면적 10㎢당 화장실"),
        x="지역",
        y="면적 10㎢당 화장실",
        title="면적 10㎢당 공중화장실 수",
        color="면적 10㎢당 화장실"
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

st.subheader("상대적으로 시설이 부족한 지역")

data["부족 후보 점수"] = (
    data["인구 1만 명당 화장실"].rank(pct=True) * 0.5 +
    data["면적 10㎢당 화장실"].rank(pct=True) * 0.5
)

shortage = data.sort_values("부족 후보 점수").head(10)

st.caption(
    "두 지표의 순위를 동일한 비중으로 결합한 탐색용 지표입니다. "
    "낮은 순위의 지역일수록 인구와 면적 대비 화장실 수가 적습니다."
)

st.dataframe(
    shortage[
        ["지역", "공중화장실수", "인구수", "면적",
         "인구 1만 명당 화장실", "면적 10㎢당 화장실"]
    ].round(2),
    use_container_width=True,
    hide_index=True
)

# 회귀분석
st.divider()
st.header("5. 회귀분석: 인구와 면적으로 화장실 수 예측")

X = data[["인구수", "면적"]]
y = data["공중화장실수"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

models = {
    "평균 모델": None,
    "선형회귀": LinearRegression(),
    "랜덤포레스트": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
}

results = []
predictions = {}

for name, model in models.items():
    if model is None:
        pred = np.repeat(y_train.mean(), len(y_test))
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

    pred = np.maximum(pred, 0)
    predictions[name] = pred

    results.append({
        "모델": name,
        "MAE": mean_absolute_error(y_test, pred),
        "MSE": mean_squared_error(y_test, pred),
        "R²": r2_score(y_test, pred)
    })

result_df = pd.DataFrame(results)

st.subheader("모델 성능 비교")
st.dataframe(
    result_df.round(3),
    use_container_width=True,
    hide_index=True
)

best_name = result_df.sort_values("MAE").iloc[0]["모델"]
st.info(f"테스트 데이터에서 MAE가 가장 낮은 모델: {best_name}")

selected_model = st.selectbox(
    "예측 결과를 확인할 모델",
    ["선형회귀", "랜덤포레스트"]
)

pred = predictions[selected_model]

compare = pd.DataFrame({
    "실제 화장실 수": y_test.values,
    "예측 화장실 수": pred
})

fig = px.scatter(
    compare,
    x="실제 화장실 수",
    y="예측 화장실 수",
    title=f"{selected_model}: 실제값과 예측값 비교",
    trendline="ols"
)
st.plotly_chart(fig, use_container_width=True)

# 전체 지역 예측
if selected_model == "선형회귀":
    final_model = LinearRegression()
else:
    final_model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

final_model.fit(X, y)
data["예측 화장실 수"] = np.maximum(final_model.predict(X), 0)
data["실제-예측 차이"] = (
    data["공중화장실수"] - data["예측 화장실 수"]
)

st.subheader("실제 화장실 수와 예측 화장실 수")
st.caption("예측값보다 실제 개수가 적은 지역은 추가 조사가 필요한 후보입니다.")

st.dataframe(
    data.sort_values("실제-예측 차이")[
        ["지역", "공중화장실수", "예측 화장실 수", "실제-예측 차이"]
    ].round(2),
    use_container_width=True,
    hide_index=True
)

# 군집분석
st.divider()
st.header("6. 군집분석: 비슷한 지역끼리 분류")

features = data[
    ["공중화장실수", "인구 1만 명당 화장실", "면적 10㎢당 화장실"]
].copy()

scaler = StandardScaler()
scaled = scaler.fit_transform(features)

k = st.slider(
    "군집 수 선택",
    min_value=2,
    max_value=min(7, len(data) - 1),
    value=min(3, len(data) - 1)
)

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

data["군집"] = kmeans.fit_predict(scaled) + 1

if k < len(data):
    score = silhouette_score(scaled, data["군집"])
    st.metric("실루엣 점수", f"{score:.3f}")
    st.caption("실루엣 점수는 군집이 얼마나 잘 구분되는지 보여 줍니다. 1에 가까울수록 군집 구분이 뚜렷합니다.")

fig = px.scatter(
    data,
    x="인구 1만 명당 화장실",
    y="면적 10㎢당 화장실",
    color=data["군집"].astype(str),
    hover_name="지역",
    size="공중화장실수",
    title="지역별 공중화장실 특성 군집",
    labels={
        "color": "군집",
        "인구 1만 명당 화장실": "인구 1만 명당 화장실 수",
        "면적 10㎢당 화장실": "면적 10㎢당 화장실 수"
    }
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("군집별 평균 특성")

cluster_summary = data.groupby("군집").agg(
    지역수=("지역", "count"),
    평균화장실수=("공중화장실수", "mean"),
    평균인구=("인구수", "mean"),
    평균면적=("면적", "mean"),
    인구대비시설=("인구 1만 명당 화장실", "mean"),
    면적대비시설=("면적 10㎢당 화장실", "mean")
).reset_index()

st.dataframe(
    cluster_summary.round(2),
    use_container_width=True,
    hide_index=True
)

st.subheader("군집별 지역 목록")

for cluster in sorted(data["군집"].unique()):
    with st.expander(f"군집 {cluster}에 속한 지역"):
        st.write(
            ", ".join(
                data.loc[data["군집"] == cluster, "지역"].tolist()
            )
        )

# 최종 데이터
st.divider()
st.header("7. 전체 분석 데이터")

st.dataframe(
    data.sort_values("부족 후보 점수"),
    use_container_width=True,
    hide_index=True
)

csv = data.to_csv(index=False, encoding="utf-8-sig")

st.download_button(
    "📥 분석 결과 CSV 다운로드",
    data=csv,
    file_name="공중화장실_지역별_분석결과.csv",
    mime="text/csv"
)

st.divider()

st.caption("""
분석 주의사항:
이 분석은 공중화장실의 개수, 주민등록인구, 행정구역 면적을 활용한
지역 단위 비교입니다. 실제 이동거리, 관광객, 유동인구, 운영시간,
시설의 개방 여부 등은 충분히 반영하지 못하므로
시설 부족을 확정하는 결과가 아닌 추가 조사가 필요한 지역을 찾는 자료로 활용해야 합니다.
""")
