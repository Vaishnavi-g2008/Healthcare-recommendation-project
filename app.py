import pandas as pd
import streamlit as st
import pickle

st.title("Healthcare Recommendation System")

# Load Dataset
df = pd.read_csv("cleaned_healthcare_dataset.csv")

# Load Similarity Matrix
with open("similarity.pkl", "rb") as file:
    similarities = pickle.load(file)

# Hospital List
hospitals = df["Hospital"].tolist()

# Dropdown
name = st.selectbox("Select Hospital", hospitals)


# Get Hospital Name by Index
def get_hospital_by_index(i):
    if i >= 0 and i < len(df):
        return df.loc[i, "Hospital"]
    return ""


# Get Index from Hospital Name
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


# Recommendation Button
if st.button("Recommend"):

    index = get_index_from_hospital(name)

    if index == -1:
        st.error("Hospital Not Found")

    else:

        similarity_indexes = list(enumerate(similarities[index]))

        similarity_indexes = sorted(
            similarity_indexes,
            key=lambda x: x[1],
            reverse=True
        )

        st.subheader("Recommended Hospitals")

        count = 0

        for i in similarity_indexes[1:]:

            hospital = get_hospital_by_index(i[0])

            if hospital != "":
                st.write(f"{count+1}. {hospital}")
                count += 1

            if count == 5:
                break