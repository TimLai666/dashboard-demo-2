from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


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

    filtered = df[
        (df["order_date"].dt.date >= start_date)
        & (df["order_date"].dt.date <= end_date)
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
    st.caption("以 uv + Streamlit 建立的第一版儀表板")

    data = load_data()
    model_df = build_model(data)
    filtered = sidebar_filters(model_df)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["總覽", "商品", "地區", "客戶", "資料品質", "資料預覽"]
    )

    with tab1:
        render_overview(filtered)
    with tab2:
        render_product(filtered)
    with tab3:
        render_region(filtered)
    with tab4:
        render_customer(filtered)
    with tab5:
        render_quality(data, model_df)
    with tab6:
        render_data_preview(data, model_df, filtered)


if __name__ == "__main__":
    main()
