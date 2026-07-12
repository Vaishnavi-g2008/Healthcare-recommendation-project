import pandas as pd
import streamlit as st
import pickle
import time

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="Healthcare Recommendation System",
    page_icon="🏥",
    layout="centered"
)

# ---------------- Sidebar ----------------
st.sidebar.title("🏥 Healthcare Recommendation")
st.sidebar.info(
    """
    This application recommends similar hospitals
    using **NLP**, **TF-IDF**, and **Cosine Similarity**.
    """
)

# ---------------- Title ----------------
st.title("🏥 Healthcare Recommendation System")
st.markdown("### Find Similar Hospitals Using NLP")
st.write("Select a hospital from the list and click **Recommend Hospitals**.")

st.divider()

# ---------------- Load Dataset ----------------
df = pd.read_csv("cleaned_healthcare_dataset.csv")

# ---------------- Load Similarity ----------------
with open("similarity.pkl", "rb") as file:
    similarities = pickle.load(file)

# ---------------- Metrics ----------------
col1, col2 = st.columns(2)

with col1:
    st.metric("🏥 Total Hospitals", df["Hospital"].nunique())

with col2:
    st.metric("📄 Total Records", len(df))

st.divider()

# ---------------- Hospital List ----------------
hospitals = sorted(df["Hospital"].unique())

# ---------------- Dropdown ----------------
name = st.selectbox("🏥 Select Hospital", hospitals)

# ---------------- Get Hospital Name ----------------
def get_hospital_by_index(i):
    if i >= 0 and i < len(df):
        return df.loc[i, "Hospital"]
    return ""

# ---------------- Get Hospital Index ----------------
def get_index_from_hospital(name):

    clean_name = name.strip().lower().replace(" ", "").replace("-", "")

    match = df[
        df["Hospital"]
        .str.lower()
        .str.replace(" ", "", regex=False)
        .str.replace("-", "", regex=False)
        == clean_name
    ]

    if not match.empty:
        return match.index[0]

    return -1

# ---------------- Recommendation Button ----------------
if st.button("🔍 Recommend Hospitals"):

    with st.spinner("Finding similar hospitals..."):

        time.sleep(1)

        index = get_index_from_hospital(name)

        if index == -1:

            st.error("❌ Hospital Not Found")

        else:

            st.success(f"✅ Selected Hospital: {name}")

            similarity_indexes = list(enumerate(similarities[index]))

            similarity_indexes = sorted(
                similarity_indexes,
                key=lambda x: x[1],
                reverse=True
            )

            st.subheader("🏥 Top 5 Recommended Hospitals")

            count = 0

            for i in similarity_indexes[1:]:

                hospital = get_hospital_by_index(i[0])

                if hospital != "":
                    st.markdown(
                        f"""
                        <div style="
                        background-color:#f0f8ff;
                        padding:12px;
                        border-radius:10px;
                        margin-bottom:10px;
                        border-left:6px solid #1f77b4;">
                        <b>🏥 {count+1}. {hospital}</b>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    count += 1

                if count == 5:
                    break

            st.balloons()

st.divider()

st.markdown(
    """
    <center>
    ❤️ <b>Healthcare Recommendation System</b><br>
    Developed using <b>Python | Streamlit | NLP | TF-IDF | Cosine Similarity</b>
    </center>
    """,
    unsafe_allow_html=True
)
