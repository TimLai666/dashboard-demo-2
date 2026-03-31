from pathlib import Path
from itertools import combinations

import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import f_oneway, ttest_ind


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


@st.cache_data(show_spinner=False)
def load_data() -> dict[str, pd.DataFrame]:
    sales = pd.read_csv(
        DATA_DIR / "sales.csv",
        parse_dates=["order_date"],
        low_memory=False,
    )
    customers = pd.read_csv(DATA_DIR / "customers.csv", parse_dates=["join_date"])
    products = pd.read_csv(DATA_DIR / "products.csv")
    stores = pd.read_csv(DATA_DIR / "stores.csv")
    calendar = pd.read_csv(DATA_DIR / "calendar.csv", parse_dates=["date"])

    return {
        "sales": sales,
        "customers": customers,
        "products": products,
        "stores": stores,
        "calendar": calendar,
    }


def build_model(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = data["sales"].merge(data["products"], on="product_id", how="left")
    df = df.merge(data["stores"], on="store_id", how="left")
    df = df.merge(data["customers"], on="customer_id", how="left")
    df["month"] = df["order_date"].dt.to_period("M").astype(str)
    df["age_group"] = pd.cut(
        df["age"],
        bins=[17, 24, 34, 44, 54, 100],
        labels=["18-24", "25-34", "35-44", "45-54", "55+"],
    )

    for col in ["country", "brand", "category", "store_type", "city", "gender"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("篩選條件")

    min_date = df["order_date"].min().date()
    max_date = df["order_date"].max().date()

    default_countries = sorted(df["country"].dropna().unique().tolist())
    default_brands = sorted(df["brand"].dropna().unique().tolist())
    default_categories = sorted(df["category"].dropna().unique().tolist())

    if "filter_date_range" not in st.session_state:
        st.session_state["filter_date_range"] = (min_date, max_date)
    if "filter_countries" not in st.session_state:
        st.session_state["filter_countries"] = default_countries
    if "filter_brands" not in st.session_state:
        st.session_state["filter_brands"] = default_brands
    if "filter_categories" not in st.session_state:
        st.session_state["filter_categories"] = default_categories
    if "filter_member" not in st.session_state:
        st.session_state["filter_member"] = "全部"

    if st.sidebar.button("重置篩選器", use_container_width=True):
        st.session_state["filter_date_range"] = (min_date, max_date)
        st.session_state["filter_countries"] = default_countries
        st.session_state["filter_brands"] = default_brands
        st.session_state["filter_categories"] = default_categories
        st.session_state["filter_member"] = "全部"
        st.rerun()

    date_range = st.sidebar.date_input(
        "日期區間",
        min_value=min_date,
        max_value=max_date,
        key="filter_date_range",
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = min_date
        end_date = max_date

    countries = default_countries
    brands = default_brands
    categories = default_categories
    member_values = ["全部", "會員", "非會員"]

    selected_countries = st.sidebar.multiselect("國家", countries, key="filter_countries")
    selected_brands = st.sidebar.multiselect("品牌", brands, key="filter_brands")
    selected_categories = st.sidebar.multiselect(
        "品類",
        categories,
        key="filter_categories",
    )
    selected_member = st.sidebar.selectbox(
        "會員狀態", member_values, key="filter_member"
    )

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    filtered = df[
        df["order_date"].between(start_ts, end_ts)
        & (df["country"].isin(selected_countries))
        & (df["brand"].isin(selected_brands))
        & (df["category"].isin(selected_categories))
    ]

    if selected_member == "會員":
        filtered = filtered[filtered["loyalty_member"] == 1]
    elif selected_member == "非會員":
        filtered = filtered[filtered["loyalty_member"] == 0]

    return filtered


def render_overview(df: pd.DataFrame) -> None:
    total_revenue = float(df["revenue"].sum())
    total_profit = float(df["profit"].sum())
    total_orders = int(df["order_id"].nunique())
    total_customers = int(df["customer_id"].nunique())
    avg_order_value = total_revenue / total_orders if total_orders else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("總營收", f"{total_revenue:,.0f}")
    c2.metric("總毛利", f"{total_profit:,.0f}")
    c3.metric("訂單數", f"{total_orders:,}")
    c4.metric("活躍客數", f"{total_customers:,}")
    c5.metric("平均客單價", f"{avg_order_value:,.2f}")

    daily = df.groupby("order_date", as_index=False)[["revenue", "profit"]].sum()
    fig = px.line(
        daily,
        x="order_date",
        y=["revenue", "profit"],
        title="營收與毛利趨勢",
        markers=True,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_product(df: pd.DataFrame) -> None:
    by_brand = df.groupby("brand", as_index=False)["revenue"].sum().sort_values(
        "revenue", ascending=False
    )
    by_category = df.groupby("category", as_index=False)["revenue"].sum().sort_values(
        "revenue", ascending=False
    )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            px.bar(by_brand, x="brand", y="revenue", title="品牌營收排名"),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            px.pie(by_category, names="category", values="revenue", title="品類營收占比"),
            use_container_width=True,
        )


def render_region(df: pd.DataFrame) -> None:
    by_country = df.groupby("country", as_index=False)["revenue"].sum().sort_values(
        "revenue", ascending=False
    )
    by_store_type = df.groupby("store_type", as_index=False)["revenue"].sum().sort_values(
        "revenue", ascending=False
    )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            px.bar(by_country, x="country", y="revenue", title="各國營收"),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            px.bar(by_store_type, x="store_type", y="revenue", title="店型營收"),
            use_container_width=True,
        )


def render_customer(df: pd.DataFrame) -> None:
    member = (
        df.assign(member=df["loyalty_member"].map({1: "會員", 0: "非會員"}))
        .groupby("member", as_index=False)["revenue"]
        .sum()
    )
    by_age = df.groupby("age_group", as_index=False)["revenue"].sum()

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            px.pie(member, names="member", values="revenue", title="會員營收占比"),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            px.bar(by_age, x="age_group", y="revenue", title="年齡層營收"),
            use_container_width=True,
        )


def _rfm_score(series: pd.Series, reverse: bool = False) -> pd.Series:
    ranked = series.rank(method="first")
    labels = [5, 4, 3, 2, 1] if reverse else [1, 2, 3, 4, 5]
    try:
        return pd.qcut(ranked, q=5, labels=labels).astype(int)
    except ValueError:
        pct = ranked.rank(method="first", pct=True)
        return pd.cut(
            pct,
            bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=labels,
            include_lowest=True,
        ).astype(int)


def calculate_customer_rfm(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "customer_id",
                "recency",
                "frequency",
                "monetary",
                "r_score",
                "f_score",
                "m_score",
                "segment",
            ]
        )

    ref_date = df["order_date"].max() + pd.Timedelta(days=1)
    rfm = (
        df.groupby("customer_id", as_index=False)
        .agg(
            last_order=("order_date", "max"),
            frequency=("order_id", "nunique"),
            monetary=("revenue", "sum"),
        )
    )
    rfm["recency"] = (ref_date - rfm["last_order"]).dt.days

    rfm["r_score"] = _rfm_score(rfm["recency"], reverse=True)
    rfm["f_score"] = _rfm_score(rfm["frequency"])
    rfm["m_score"] = _rfm_score(rfm["monetary"])

    def _segment(row: pd.Series) -> str:
        r, f, m = int(row["r_score"]), int(row["f_score"]), int(row["m_score"])
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        if r >= 3 and f >= 3 and m >= 3:
            return "Loyal"
        if r <= 2 and f >= 3:
            return "At Risk"
        if r >= 4 and f <= 2:
            return "New"
        return "Regular"

    rfm["segment"] = rfm.apply(_segment, axis=1)

    return rfm[
        [
            "customer_id",
            "recency",
            "frequency",
            "monetary",
            "r_score",
            "f_score",
            "m_score",
            "segment",
        ]
    ]


def calculate_customer_cai(df: pd.DataFrame) -> pd.DataFrame:
    work = df[["customer_id", "order_date"]].dropna().copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                "customer_id",
                "interval_count",
                "avg_interval",
                "weighted_avg_interval",
                "cai",
            ]
        )

    work["order_date"] = work["order_date"].dt.normalize()
    work = work.drop_duplicates(["customer_id", "order_date"]).sort_values(
        ["customer_id", "order_date"]
    )
    work["interval"] = work.groupby("customer_id")["order_date"].diff().dt.days.astype("float64")
    work = work.dropna(subset=["interval"])

    if work.empty:
        return pd.DataFrame(
            columns=[
                "customer_id",
                "interval_count",
                "avg_interval",
                "weighted_avg_interval",
                "cai",
            ]
        )

    work["weight"] = work.groupby("customer_id").cumcount() + 1
    work["weighted_interval"] = work["interval"] * work["weight"]

    agg = (
        work.groupby("customer_id", as_index=False)
        .agg(
            interval_count=("interval", "size"),
            avg_interval=("interval", "mean"),
            weighted_sum=("weighted_interval", "sum"),
            weight_sum=("weight", "sum"),
        )
    )
    agg = agg[agg["interval_count"] >= 3].copy()

    if agg.empty:
        return pd.DataFrame(
            columns=[
                "customer_id",
                "interval_count",
                "avg_interval",
                "weighted_avg_interval",
                "cai",
            ]
        )

    agg["weighted_avg_interval"] = agg["weighted_sum"] / agg["weight_sum"]
    agg["cai"] = (
        (agg["avg_interval"] - agg["weighted_avg_interval"])
        / agg["avg_interval"].replace(0, pd.NA)
    ) * 100
    agg["cai"] = agg["cai"].fillna(0.0)

    return agg[
        [
            "customer_id",
            "interval_count",
            "avg_interval",
            "weighted_avg_interval",
            "cai",
        ]
    ]


def run_anova_with_pairwise(
    df: pd.DataFrame, group_col: str, value_col: str
) -> tuple[pd.DataFrame, str, pd.DataFrame]:
    stats_df = (
        df.groupby(group_col, as_index=False)
        .agg(
            count=(value_col, "count"),
            mean=(value_col, "mean"),
            std=(value_col, "std"),
        )
        .sort_values("mean", ascending=False)
    )

    group_samples = []
    group_names = []
    for g, grp in df.groupby(group_col):
        sample = grp[value_col].dropna().values
        if len(sample) >= 2:
            group_names.append(str(g))
            group_samples.append(sample)

    if len(group_samples) < 2:
        return stats_df, "可用群組不足（至少 2 群且每群至少 2 筆）無法執行 ANOVA。", pd.DataFrame()

    f_stat, p_value = f_oneway(*group_samples)
    summary = f"整體 ANOVA 結果：F={f_stat:.4f}, p-value={p_value:.6g}"

    pair_rows = []
    pair_count = len(group_names) * (len(group_names) - 1) / 2
    for g1, g2 in combinations(group_names, 2):
        s1 = df.loc[df[group_col].astype(str) == g1, value_col].dropna().values
        s2 = df.loc[df[group_col].astype(str) == g2, value_col].dropna().values
        t_stat, p_raw = ttest_ind(s1, s2, equal_var=False)
        p_adj = min(p_raw * pair_count, 1.0)
        pair_rows.append(
            {
                "group_1": g1,
                "group_2": g2,
                "mean_diff": float(s1.mean() - s2.mean()),
                "t_stat": float(t_stat),
                "p_raw": float(p_raw),
                "p_adj_bonferroni": float(p_adj),
                "significant_p<0.05": bool(p_adj < 0.05),
            }
        )

    pairwise_df = pd.DataFrame(pair_rows).sort_values(
        ["significant_p<0.05", "p_adj_bonferroni"],
        ascending=[False, True],
    )
    return stats_df, summary, pairwise_df


def build_customer_metrics(df: pd.DataFrame) -> pd.DataFrame:
    spend = df.groupby("customer_id", as_index=False).agg(
        monetary=("revenue", "sum"),
        frequency=("order_id", "nunique"),
    )
    profile = df.groupby("customer_id", as_index=False).agg(
        age=("age", "first"),
        age_group=("age_group", "first"),
        gender=("gender", "first"),
        loyalty_member=("loyalty_member", "first"),
        join_date=("join_date", "first"),
    )
    out = spend.merge(profile, on="customer_id", how="left")
    out["loyalty_flag"] = out["loyalty_member"].map({1: "忠誠會員", 0: "非忠誠會員"})
    return out


def render_rfm_cai(df: pd.DataFrame) -> None:
    st.subheader("RFM 與 CAI（Customer Activity Index）")
    if df.empty:
        st.warning("目前篩選條件下沒有可分析的交易資料。")
        return

    rfm = calculate_customer_rfm(df)
    cai = calculate_customer_cai(df)
    merged = rfm.merge(cai, on="customer_id", how="left")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("顧客數", f"{len(rfm):,}")
    c2.metric("可計算 CAI 顧客", f"{merged['cai'].notna().sum():,}")
    c3.metric("平均 Monetary", f"{rfm['monetary'].mean():,.2f}")
    c4.metric("平均 Frequency", f"{rfm['frequency'].mean():.2f}")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            px.scatter(
                rfm,
                x="frequency",
                y="monetary",
                color="segment",
                size="m_score",
                hover_data=["customer_id", "recency", "r_score", "f_score", "m_score"],
                title="RFM 散點圖（F vs M）",
            ),
            use_container_width=True,
        )
    with right:
        cai_plot = merged[merged["cai"].notna()]
        if cai_plot.empty:
            st.info("可計算 CAI 的顧客不足（至少 3 個購買間隔）。")
        else:
            st.plotly_chart(
                px.histogram(
                    cai_plot,
                    x="cai",
                    nbins=30,
                    color="segment",
                    title="CAI 分布",
                ),
                use_container_width=True,
            )

    st.subheader("ANOVA：RFM 各群銷售金額差異（Monetary）")
    stats_df, summary, pairwise_df = run_anova_with_pairwise(rfm, "segment", "monetary")
    st.dataframe(stats_df, use_container_width=True)
    st.write(summary)
    if not pairwise_df.empty:
        st.dataframe(pairwise_df, use_container_width=True)

    st.subheader("ANOVA：RFM 各群消費頻率差異（Frequency）")
    freq_stats, freq_summary, freq_pairwise = run_anova_with_pairwise(rfm, "segment", "frequency")
    st.dataframe(freq_stats, use_container_width=True)
    st.write(freq_summary)
    if not freq_pairwise.empty:
        st.dataframe(freq_pairwise, use_container_width=True)


def render_demographic_significance(df: pd.DataFrame) -> None:
    st.subheader("客群顯著性：年齡、性別、忠誠會員對消費金額")
    if df.empty:
        st.warning("目前篩選條件下沒有可分析的交易資料。")
        return

    customer_metrics = build_customer_metrics(df)

    st.plotly_chart(
        px.box(customer_metrics, x="age_group", y="monetary", title="年齡層 vs 消費金額"),
        use_container_width=True,
    )
    st.plotly_chart(
        px.box(customer_metrics, x="gender", y="monetary", title="性別 vs 消費金額"),
        use_container_width=True,
    )
    st.plotly_chart(
        px.box(customer_metrics, x="loyalty_flag", y="monetary", title="是否忠誠會員 vs 消費金額"),
        use_container_width=True,
    )

    for group_col, title in [
        ("age_group", "年齡層"),
        ("gender", "性別"),
        ("loyalty_flag", "忠誠會員"),
    ]:
        st.markdown(f"**ANOVA：{title} 對消費金額**")
        stats_df, summary, pairwise_df = run_anova_with_pairwise(customer_metrics, group_col, "monetary")
        st.dataframe(stats_df, use_container_width=True)
        st.write(summary)
        if not pairwise_df.empty:
            st.dataframe(pairwise_df, use_container_width=True)


def render_city_brand_map(df: pd.DataFrame) -> None:
    st.subheader("城市 x 品牌銷售占比地圖")
    if df.empty:
        st.warning("目前篩選條件下沒有可分析的交易資料。")
        return

    city_brand = (
        df.groupby(["city", "country", "brand"], as_index=False)
        .agg(revenue=("revenue", "sum"))
    )
    city_total = df.groupby(["city", "country"], as_index=False).agg(city_revenue=("revenue", "sum"))
    city_brand = city_brand.merge(city_total, on=["city", "country"], how="left")
    city_brand["share_pct"] = city_brand["revenue"] / city_brand["city_revenue"] * 100

    coord_map = {
        "New York": (40.7128, -74.0060),
        "Melbourne": (-37.8136, 144.9631),
        "Berlin": (52.5200, 13.4050),
        "Paris": (48.8566, 2.3522),
        "Sydney": (-33.8688, 151.2093),
        "Toronto": (43.6532, -79.3832),
        "London": (51.5074, -0.1278),
    }
    city_brand["lat"] = city_brand["city"].map(lambda c: coord_map.get(c, (None, None))[0])
    city_brand["lon"] = city_brand["city"].map(lambda c: coord_map.get(c, (None, None))[1])
    city_brand = city_brand.dropna(subset=["lat", "lon"])

    brand_options = sorted(city_brand["brand"].unique().tolist())
    selected_brand = st.selectbox("地圖品牌", brand_options)
    brand_map = city_brand[city_brand["brand"] == selected_brand]

    st.plotly_chart(
        px.scatter_geo(
            brand_map,
            lat="lat",
            lon="lon",
            size="share_pct",
            color="share_pct",
            hover_name="city",
            hover_data={"country": True, "revenue": ":.2f", "share_pct": ":.2f"},
            title=f"{selected_brand} 在各城市銷售占比(%)",
        ),
        use_container_width=True,
    )

    share_heat = city_brand.pivot_table(
        index="city",
        columns="brand",
        values="share_pct",
        aggfunc="mean",
        fill_value=0,
    )
    st.plotly_chart(
        px.imshow(
            share_heat,
            labels={"x": "品牌", "y": "城市", "color": "占比(%)"},
            title="各城市各品牌銷售占比熱圖",
            aspect="auto",
        ),
        use_container_width=True,
    )


def render_member_trend(customers_df: pd.DataFrame) -> None:
    st.subheader("加入會員時間趨勢與忠誠化比例")
    if customers_df.empty:
        st.warning("沒有可分析的會員資料。")
        return

    cdf = customers_df.copy()
    cdf = cdf.dropna(subset=["join_date"])
    cdf = cdf[cdf["join_date"] <= pd.Timestamp.today()]
    if cdf.empty:
        st.warning("可用的 join_date 資料不足。")
        return

    cdf["join_month"] = cdf["join_date"].dt.to_period("M").astype(str)
    trend = (
        cdf.groupby("join_month", as_index=False)
        .agg(
            joiners=("customer_id", "count"),
            loyalty_rate=("loyalty_member", "mean"),
        )
    )
    trend["loyalty_rate_pct"] = trend["loyalty_rate"] * 100

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            px.line(trend, x="join_month", y="joiners", markers=True, title="每月加入會員人數"),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            px.line(
                trend,
                x="join_month",
                y="loyalty_rate_pct",
                markers=True,
                title="不同加入月份成為忠誠會員比例(%)",
            ),
            use_container_width=True,
        )


def render_quality(data: dict[str, pd.DataFrame], model_df: pd.DataFrame) -> None:
    sales = data["sales"]
    quality = pd.DataFrame(
        {
            "table": ["sales", "customers", "products", "stores", "calendar"],
            "rows": [
                len(data["sales"]),
                len(data["customers"]),
                len(data["products"]),
                len(data["stores"]),
                len(data["calendar"]),
            ],
            "missing_rate": [
                float(data["sales"].isna().mean().mean()),
                float(data["customers"].isna().mean().mean()),
                float(data["products"].isna().mean().mean()),
                float(data["stores"].isna().mean().mean()),
                float(data["calendar"].isna().mean().mean()),
            ],
        }
    )

    fk_missing = pd.DataFrame(
        {
            "check": [
                "product_id 對應率",
                "store_id 對應率",
                "customer_id 對應率",
            ],
            "matched_rate": [
                model_df["product_name"].notna().mean(),
                model_df["store_name"].notna().mean(),
                model_df["age"].notna().mean(),
            ],
        }
    )

    anomaly = pd.DataFrame(
        {
            "rule": [
                "quantity <= 0",
                "revenue < 0",
                "profit + cost 與 revenue 不一致(四捨五入誤差 0.01)",
            ],
            "count": [
                int((sales["quantity"] <= 0).sum()),
                int((sales["revenue"] < 0).sum()),
                int((sales["profit"] + sales["cost"] - sales["revenue"]).abs().gt(0.01).sum()),
            ],
        }
    )

    st.subheader("資料表概況")
    st.dataframe(quality, use_container_width=True)
    st.subheader("外鍵對應率")
    st.dataframe(fk_missing, use_container_width=True)
    st.subheader("異常規則檢查")
    st.dataframe(anomaly, use_container_width=True)


def render_data_preview(
    data: dict[str, pd.DataFrame], model_df: pd.DataFrame, filtered_df: pd.DataFrame
) -> None:
    st.subheader("資料表預覽")
    apply_filter = st.toggle("套用目前篩選", value=True)
    table_name = st.selectbox("選擇資料表", ["sales", "customers", "products", "stores", "calendar", "model"])

    if apply_filter:
        filtered_order_ids = set(filtered_df["order_id"].dropna().unique().tolist())
        filtered_customer_ids = set(filtered_df["customer_id"].dropna().unique().tolist())
        filtered_product_ids = set(filtered_df["product_id"].dropna().unique().tolist())
        filtered_store_ids = set(filtered_df["store_id"].dropna().unique().tolist())
        filtered_dates = set(filtered_df["order_date"].dt.normalize().dropna().tolist())

        if table_name == "model":
            source = filtered_df
        elif table_name == "sales":
            source = data["sales"][data["sales"]["order_id"].isin(filtered_order_ids)]
        elif table_name == "customers":
            source = data["customers"][data["customers"]["customer_id"].isin(filtered_customer_ids)]
        elif table_name == "products":
            source = data["products"][data["products"]["product_id"].isin(filtered_product_ids)]
        elif table_name == "stores":
            source = data["stores"][data["stores"]["store_id"].isin(filtered_store_ids)]
        else:
            source = data["calendar"][data["calendar"]["date"].dt.normalize().isin(filtered_dates)]
    else:
        source = model_df if table_name == "model" else data[table_name]

    st.write(f"列數: {len(source):,} | 欄位數: {len(source.columns)}")
    st.dataframe(source.head(100), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Sales Dashboard", layout="wide")
    st.title("Sales Dashboard Demo")
    st.caption("以 uv + Streamlit 建立的主題式多頁儀表板")

    data = load_data()
    if "model_df" not in st.session_state:
        with st.spinner("初始化資料模型中，首次載入可能需要一些時間..."):
            st.session_state["model_df"] = build_model(data)
    model_df = st.session_state["model_df"]
    filtered = sidebar_filters(model_df)

    st.sidebar.divider()
    st.sidebar.subheader("主題分頁")
    page = st.sidebar.radio(
        "選擇頁面",
        [
            "總覽",
            "RFM與CAI",
            "客群顯著性",
            "城市品牌地圖",
            "會員趨勢",
            "資料品質與預覽",
        ],
    )

    if page == "總覽":
        render_overview(filtered)
        st.divider()
        render_product(filtered)
        st.divider()
        render_region(filtered)
        st.divider()
        render_customer(filtered)
    elif page == "RFM與CAI":
        render_rfm_cai(filtered)
    elif page == "客群顯著性":
        render_demographic_significance(filtered)
    elif page == "城市品牌地圖":
        render_city_brand_map(filtered)
    elif page == "會員趨勢":
        render_member_trend(data["customers"])
    else:
        render_quality(data, model_df)
        st.divider()
        render_data_preview(data, model_df, filtered)


if __name__ == "__main__":
    main()
