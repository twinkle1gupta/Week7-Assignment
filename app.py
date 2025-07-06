import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🌸 Iris Flower Species Predictor")
st.write("""
This Streamlit app predicts the species of an Iris flower based on user input features.
""")

st.sidebar.header("Input Features")
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length (cm)', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width (cm)', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length (cm)', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width (cm)', 0.1, 2.5, 0.2)
    return pd.DataFrame({
        'sepal length (cm)': [sepal_length],
        'sepal width (cm)': [sepal_width],
        'petal length (cm)': [petal_length],
        'petal width (cm)': [petal_width],
    })

df = user_input_features()

st.subheader("Input Features")
st.write(df)

# Make predictions
prediction = model.predict(df)
proba = model.predict_proba(df)

species = ['setosa', 'versicolor', 'virginica']

st.subheader("Prediction")
st.write(f"**Predicted species:** {species[prediction[0]]}")

st.subheader("Prediction Probabilities")
st.dataframe(pd.DataFrame(proba, columns=species))

# Bar chart
st.subheader("Prediction Probability Chart")
fig, ax = plt.subplots()
sns.barplot(data=pd.DataFrame(proba, columns=species), ax=ax, palette="viridis")
ax.set_ylim(0, 1)
st.pyplot(fig)
