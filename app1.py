import streamlit as st
import pandas as pd
import time

# ---------------- Page Config ----------------

st.set_page_config(
    page_title="Healthcare Recommendation System",
    page_icon="🏥",
    layout="centered"
)

# ---------------- Load Dataset ----------------

df = pd.read_csv("cleaned_healthcare_dataset.csv")
df = df.reset_index(drop=True)

# ---------------- Sidebar ----------------

st.sidebar.title("🏥 Healthcare Recommendation")
st.sidebar.info("""
This application helps users find hospitals
based on Medical Conditions.
""")

# ---------------- Title ----------------

st.title("🏥 Healthcare Recommendation System")

st.write(
    "Select a Medical Condition and get recommended hospitals."
)

st.divider()

# ---------------- Metrics ----------------

c1, c2 = st.columns(2)

with c1:
    st.metric("Hospitals", df["Hospital"].nunique())

with c2:
    st.metric("Records", len(df))

st.divider()

# ---------------- Disease Dropdown ----------------

disease = st.selectbox(
    "🩺 Select Medical Condition",
    sorted(df["Medical Condition"].dropna().unique())
)

# ---------------- Recommendation ----------------

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
                    unsafe_allow_html=True,
                )

            st.balloons()

st.divider()

st.markdown(
"""
### ❤️ Healthcare Recommendation System

Developed using **Python | Streamlit | NLP**
"""
) 