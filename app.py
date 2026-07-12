import streamlit as st
import pandas as pd
import pickle
import time

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Healthcare Recommendation System",
    page_icon="🏥",
    layout="centered"
)

# ---------------- Load Data ----------------
df = pd.read_csv("cleaned_healthcare_dataset.csv")

# Match dataset size with similarity matrix if needed
df = df.reset_index(drop=True)

with open("similarity.pkl", "rb") as f:
    similarity = pickle.load(f)

# Keep only rows that exist in similarity matrix
df = df.iloc[:len(similarity)].reset_index(drop=True)

# ---------------- Sidebar ----------------
st.sidebar.title("🏥 Healthcare Recommendation")
st.sidebar.info(
"""
This project recommends similar hospitals using

✅ NLP

✅ TF-IDF

✅ Cosine Similarity
"""
)

# ---------------- Title ----------------
st.title("🏥 Healthcare Recommendation System")

st.write(
    "Select a hospital and get the top 5 similar hospital recommendations."
)

st.divider()

# ---------------- Metrics ----------------
c1, c2 = st.columns(2)

with c1:
    st.metric("Hospitals", df["Hospital"].nunique())

with c2:
    st.metric("Records", len(df))

st.divider()

# ---------------- Dropdown ----------------
hospital_name = st.selectbox(
    "Select Hospital",
    sorted(df["Hospital"].unique())
)

# ---------------- Recommend ----------------
if st.button("🔍 Recommend Hospitals"):

    with st.spinner("Finding similar hospitals..."):

        time.sleep(1)

        index = df[df["Hospital"] == hospital_name].index[0]

        similarity_scores = list(enumerate(similarity[index]))

        similarity_scores = sorted(
            similarity_scores,
            key=lambda x: x[1],
            reverse=True
        )

        st.success(f"Selected Hospital: {hospital_name}")

        st.subheader("🏥 Recommended Hospitals")

        count = 1

        for i in similarity_scores[1:]:

            hospital = df.iloc[i[0]]["Hospital"]

            st.markdown(
                f"""
<div style="
background:#f8fbff;
padding:12px;
border-radius:10px;
margin-bottom:10px;
border-left:6px solid #0d6efd;">
<b>{count}. 🏥 {hospital}</b>
</div>
""",
                unsafe_allow_html=True
            )

            count += 1

            if count == 6:
                break

        st.balloons()

st.divider()

st.markdown(
"""
<center>

### ❤️ Healthcare Recommendation System

Python | Streamlit | NLP | TF-IDF | Cosine Similarity

</center>
""",
unsafe_allow_html=True
)
