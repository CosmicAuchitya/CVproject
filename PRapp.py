import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import base64
from pathlib import Path

# ------------------------------------------------------------------
# 0. Page config (must be the first Streamlit command)
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Beta Product Recommendation Project",
    page_icon="🧠",
    layout="wide"
)


def set_background(image_path: str):
    """
    Set an image as the full-page background in Streamlit and
    make content readable by adding light panels on top.
    """
    img_path = Path(image_path)

    if not img_path.exists():
        st.warning(f"Background image not found: {img_path}")
        return

    with open(img_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    css = f"""
    <style>
    /* Background image on the whole app */
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }}

    /* Sidebar panel */
    [data-testid="stSidebar"] > div:first-child {{
        background-color: rgba(12, 17, 23, 0.98);
        backdrop-filter: blur(4px);
        color: #f5f7fa;
    }}

    /* Sidebar text */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] p {{
        color: #e5ecff !important;
        font-weight: 500;
    }}

    /* Dropdown text color (sidebar) */
    [data-baseweb="select"] div {{
        color: #0b1220 !important;
    }}

    /* Main content panel */
    .main .block-container {{
        background-color: rgba(255, 255, 255, 0.92);
        backdrop-filter: blur(4px);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }}

    /* Headings */
    h1, h2, h3, h4, h5, h6 {{
        color: #102A43;
    }}

    /* Normal text */
    .stMarkdown, p, span, div {{
        color: #102A43;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# Call once, right after page_config
set_background("bg_image.png")


# ------------------------------------------------------------------
# 1. Data loading (local sample CSV)
# ------------------------------------------------------------------

SAMPLE_FILE_NAME = "sample_cleaned_products.csv"


def _postprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Common cleaning steps for the sample CSV.
    """
    # Basic numeric cleaning
    numeric_cols = ["discount_price", "actual_price", "ratings", "no_of_ratings"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure main text fields exist
    for col in ["name", "main_category", "sub_category"]:
        if col not in df.columns:
            df[col] = ""

    # Drop rows without a name
    df = df[df["name"].notna()]

    return df


@st.cache_data(show_spinner=True)
def load_products() -> pd.DataFrame:
    """
    Load 'sample_cleaned_products.csv' from the local folder.
    This is a smaller sample just to demonstrate that the app works.
    """
    path = Path(SAMPLE_FILE_NAME)

    if not path.exists():
        raise RuntimeError(
            f"Sample file '{SAMPLE_FILE_NAME}' not found. "
            f"Please place it in the same folder as this app."
        )

    df = pd.read_csv(path)
    df = _postprocess_dataframe(df)
    return df


@st.cache_resource(show_spinner=True)
def build_tfidf_engine(df: pd.DataFrame):
    """
    Build a TF-IDF representation for product name + category fields.
    Returns:
        vectorizer: fitted TfidfVectorizer
        tfidf_matrix: sparse matrix (n_products x n_features)
    """
    corpus = (
        df["name"].fillna("") + " " +
        df["main_category"].fillna("") + " " +
        df["sub_category"].fillna("")
    )

    # Safety check: ensure there is at least some non-empty text
    non_empty = corpus.str.strip()
    non_empty = non_empty[non_empty != ""]
    if non_empty.empty:
        raise ValueError(
            "No valid text found in name/main_category/sub_category columns. "
            "Please check that the CSV loaded correctly."
        )

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        stop_words="english"
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix


def recommend_by_index(
    product_index: int,
    all_products: pd.DataFrame,
    tfidf_matrix,
    top_n: int = 10,
    use_price_rating_weights: bool = True
) -> pd.DataFrame:
    """
    Given a product index from all_products, return top_n similar products.

    Similarity is computed from TF-IDF on name + categories and optionally
    reweighted by ratings, number of ratings and discount percentage.
    """
    if product_index not in all_products.index:
        raise ValueError("Invalid product index")

    # 1) Base TF-IDF cosine similarity
    cosine_similarities = linear_kernel(
        tfidf_matrix[product_index:product_index + 1],
        tfidf_matrix
    ).flatten()
    scores = cosine_similarities.copy()

    # 2) Optional reweighting with ratings, number of ratings, discount %
    if use_price_rating_weights:
        ratings = all_products["ratings"].fillna(0).astype(float).values
        r_min, r_max = ratings.min(), ratings.max()
        ratings_norm = (ratings - r_min) / (r_max - r_min + 1e-6)

        num_ratings = all_products["no_of_ratings"].fillna(0).astype(float).values
        num_ratings_log = np.log1p(num_ratings)
        nr_min, nr_max = num_ratings_log.min(), num_ratings_log.max()
        num_ratings_norm = (num_ratings_log - nr_min) / (nr_max - nr_min + 1e-6)

        actual = all_products["actual_price"].replace(0, np.nan).astype(float)
        discount = all_products["discount_price"].astype(float)
        discount_pct = ((actual - discount) / actual).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0).values
        d_min, d_max = discount_pct.min(), discount_pct.max()
        discount_norm = (discount_pct - d_min) / (d_max - d_min + 1e-6)

        # Weight combination (no brand-level bias)
        weight = 1.0 + 0.4 * ratings_norm + 0.3 * num_ratings_norm + 0.3 * discount_norm
        scores = scores * weight

    # 3) Rank and select top N
    score_series = pd.Series(scores, index=all_products.index)
    score_series = score_series.drop(product_index, errors="ignore")
    top_indices = score_series.sort_values(ascending=False).head(top_n).index

    cols = [
        "name",
        "main_category",
        "sub_category",
        "discount_price",
        "actual_price",
        "ratings",
        "no_of_ratings",
    ]
    cols = [c for c in cols if c in all_products.columns]

    return all_products.loc[top_indices, cols]


# ------------------------------------------------------------------
# 2. Load data + build model (done once thanks to caching)
# ------------------------------------------------------------------
try:
    all_products = load_products()
    if all_products.empty:
        st.error(
            "Loaded dataset is empty. Please check that 'sample_cleaned_products.csv' "
            "has data."
        )
        st.stop()
except Exception as e:
    st.error(f"Error loading local sample data: {e}")
    st.stop()

try:
    tfidf_vectorizer, tfidf_matrix = build_tfidf_engine(all_products)
except Exception as e:
    st.error(f"Error building TF-IDF engine: {e}")
    st.stop()


# ------------------------------------------------------------------
# 3. Streamlit UI – layout
# ------------------------------------------------------------------
st.title("🛒 Beta Product Recommendation Project")

st.markdown(
    """
    This demo app uses a **sample subset** of the full product dataset
    to show content-based product recommendations using TF-IDF similarity
    with price & rating aware re-ranking.
    """
)

# Sidebar controls
st.sidebar.header("Search & Filters")

search_query = st.sidebar.text_input(
    "Search by product name",
    value="",
    help="Type part of a product name, e.g. 'Lloyd 1.5 Ton'"
)

# Main category filter
main_categories = sorted(all_products["main_category"].dropna().unique().tolist())
main_category_selected = st.sidebar.selectbox(
    "Main Category (optional)",
    options=["All"] + main_categories
)

# Sub category filter (depends on main category)
if main_category_selected != "All":
    subcat_source = all_products[
        all_products["main_category"] == main_category_selected
    ]
else:
    subcat_source = all_products

sub_categories = sorted(subcat_source["sub_category"].dropna().unique().tolist())
sub_category_selected = st.sidebar.selectbox(
    "Sub Category (optional)",
    options=["All"] + sub_categories
)

# Rating filter presets
rating_filter_label = st.sidebar.selectbox(
    "Rating filter",
    options=[
        "No filter",
        "1.0 – 3.0",
        "1.0 – 5.0",
        "2.0 – 3.0",
        "3.0 – 4.0",
        "4.0 – 5.0",
        "2.0 – 5.0",
    ],
    index=0
)

rating_ranges = {
    "1.0 – 3.0": (1.0, 3.0),
    "1.0 – 5.0": (1.0, 5.0),
    "2.0 – 3.0": (2.0, 3.0),
    "3.0 – 4.0": (3.0, 4.0),
    "4.0 – 5.0": (4.0, 5.0),
    "2.0 – 5.0": (2.0, 5.0),
}

# Price range slider (discount_price)
min_price = float(all_products["discount_price"].min(skipna=True) or 0)
max_price = float(all_products["discount_price"].max(skipna=True) or 0)

price_range = st.sidebar.slider(
    "Price range (based on discount_price)",
    min_value=int(min_price),
    max_value=int(max_price) if max_price > min_price else int(min_price + 1),
    value=(
        int(min_price),
        int(max_price) if max_price > min_price else int(min_price + 1)
    ),
    step=100
)

top_n_recs = st.sidebar.slider(
    "Number of recommendations",
    min_value=5,
    max_value=20,
    value=10,
    step=1
)

use_price_rating_weights = st.sidebar.checkbox(
    "Use price & rating weights",
    value=True,
    help="If checked, recommendations are reweighted by ratings, number of ratings and discount %."
)

# ------------------------------------------------------------------
# 4. Run search + filters
# ------------------------------------------------------------------
st.subheader("Search & Filtered Product List")

if st.button("Run search", type="primary"):
    results = all_products.copy()

    # Name search (case-insensitive substring)
    if search_query.strip():
        mask = results["name"].str.contains(
            search_query.strip(), case=False, na=False
        )
        results = results[mask]

    # Main category filter
    if main_category_selected != "All":
        results = results[results["main_category"] == main_category_selected]

    # Sub category filter
    if sub_category_selected != "All":
        results = results[results["sub_category"] == sub_category_selected]

    # Rating filter
    if rating_filter_label in rating_ranges and rating_filter_label != "No filter":
        low_r, high_r = rating_ranges[rating_filter_label]
        results = results[
            results["ratings"].notna() &
            (results["ratings"] >= low_r) &
            (results["ratings"] <= high_r)
        ]

    # Price filter
    low_p, high_p = price_range
    results = results[
        results["discount_price"].notna() &
        (results["discount_price"] >= low_p) &
        (results["discount_price"] <= high_p)
    ]

    # Save filtered results
    st.session_state["filtered_results"] = results

    if results.empty:
        st.warning("No products found for the current search + filter combination.")
    else:
        st.write(
            "Filtered results (index on the **left** is the original index from the dataset):"
        )
        cols_to_show = [
            "name",
            "main_category",
            "sub_category",
            "discount_price",
            "actual_price",
            "ratings",
            "no_of_ratings",
        ]
        cols_to_show = [c for c in cols_to_show if c in results.columns]
        st.dataframe(results[cols_to_show])
else:
    # Show last results if present
    results = st.session_state.get("filtered_results", pd.DataFrame())
    if not results.empty:
        st.write(
            "Last filtered results (index on the **left** is the original index from the dataset):"
        )
        cols_to_show = [
            "name",
            "main_category",
            "sub_category",
            "discount_price",
            "actual_price",
            "ratings",
            "no_of_ratings",
        ]
        cols_to_show = [c for c in cols_to_show if c in results.columns]
        st.dataframe(results[cols_to_show])

# ------------------------------------------------------------------
# 5. Select a product and show recommendations
# ------------------------------------------------------------------
st.subheader("Get Recommendations from a Selected Product")

filtered_results = st.session_state.get("filtered_results", pd.DataFrame())

if filtered_results is None or filtered_results.empty:
    st.info("Run a search first, then pick a product for recommendations.")
else:
    # Build options like: "272574 | Lloyd 1.5 Ton 5 Star Inverter Split AC ..."
    options = []
    for idx, row in filtered_results.iterrows():
        label = f"{idx} | {row['name'][:80]}"
        options.append((label, idx))

    labels = [o[0] for o in options]
    label_to_idx = {label: idx for label, idx in options}

    selected_label = st.selectbox(
        "Choose a product (the number before the '|' is the index used for recommendations):",
        options=labels,
        index=0 if labels else None
    )

    if st.button("Show Recommendations"):
        selected_index = label_to_idx[selected_label]

        try:
            recs = recommend_by_index(
                product_index=selected_index,
                all_products=all_products,
                tfidf_matrix=tfidf_matrix,
                top_n=top_n_recs,
                use_price_rating_weights=use_price_rating_weights
            )

            st.write(f"Recommendations similar to product index **{selected_index}**:")
            st.dataframe(recs)

        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Unexpected error while generating recommendations: {e}")
