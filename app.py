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
if st.button("🔍 Find Hospitals"):

    with st.spinner("Searching hospitals..."):

        time.sleep(1)

        hospitals = (
            df[df["Medical Condition"] == disease]["Hospital"]
            .drop_duplicates()
            .tolist()
        )

        st.success(f"✅ Selected Disease: {disease}")

        st.subheader("🏥 Recommended Hospitals")

        if len(hospitals) == 0:
            st.warning("No hospitals found.")

        else:
            for i, hospital in enumerate(hospitals[:5], start=1):

                st.markdown(
                    f"""
                    <div style="
                        background-color:#f0f8ff;
                        padding:12px;
                        border-radius:10px;
                        margin-bottom:10px;
                        border-left:6px solid #1f77b4;">
                        <b>{i}. 🏥 {hospital}</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.balloons()
        

           
