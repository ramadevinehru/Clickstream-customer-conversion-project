import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load trained models
classification_model = pickle.load(open("models/classification.pkl", "rb"))
regression_model = pickle.load(open("models/regression.pkl", "rb"))
clustering_model = pickle.load(open("models/clustering.pkl", "rb"))

# Get feature names from models
model_features = list(set(classification_model.feature_names_in_) | set(regression_model.feature_names_in_))

# Function for preprocessing user input
def preprocess_input(data, model_features):
    """Preprocess input data by handling categorical variables and ensuring consistent feature names."""
    data = data.copy()  # Avoid modifying the original DataFrame

    # Convert model_features to a list to avoid TypeError
    model_features = list(model_features)

    # Rename 'price' to 'price_2' if needed
    if 'price' in data.columns and 'price_2' in model_features:
        data.rename(columns={'price': 'price_2'}, inplace=True)

    # Identify categorical columns
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

    # Apply one-hot encoding
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True, dtype=int)

    # Standardize numerical features
    scaler = StandardScaler()
    numerical_cols = data.select_dtypes(include=['number']).columns.tolist()

    if numerical_cols:  # Ensure numerical columns exist before scaling
        data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    # Add missing columns in one step
    missing_cols = [col for col in model_features if col not in data.columns]
    if missing_cols:
        missing_data = pd.DataFrame(0, index=data.index, columns=missing_cols)
        data = pd.concat([data, missing_data], axis=1)

    # Ensure correct column order
    data = data[model_features]

    return data


# Streamlit UI
st.title("Customer Conversion Analysis for Online Shopping")

# Upload CSV or manual input
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())
else:
    st.write("Or enter data manually:")
    session_length = st.number_input("Session Length", min_value=0)
    price = st.number_input("Price", min_value=0.0)
    user_input = pd.DataFrame([[session_length, price]], columns=["session_length", "price"])
    df = user_input

# Preprocess data
df_scaled = preprocess_input(df, model_features)

# Classification Prediction
if st.button("Predict Conversion"):
    classification_pred = classification_model.predict(df_scaled)
    st.write(f"Predicted Conversion: {'Yes' if classification_pred[0] == 1 else 'No'}")

# Regression Prediction
if st.button("Estimate Revenue"):
    revenue_pred = regression_model.predict(df_scaled)
    st.write(f"Estimated Revenue: ${revenue_pred[0]:.2f}")

# Clustering Visualization
if st.button("Show Customer Segments"):
    clusters = clustering_model.predict(df_scaled)
    df['Cluster'] = clusters
    st.write("Customer Segmentation:")
    st.write(df[['Cluster']])

    # Visualize clusters
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df["session_length"], y=df["price"], hue=df["Cluster"], palette='viridis')
    plt.xlabel("Session Length")
    plt.ylabel("Price")
    plt.title("Customer Segments")
    st.pyplot(plt)

# Data Visualizations
st.write("### Data Visualizations")
if st.button("Show Distribution"):
    plt.figure(figsize=(8, 6))
    sns.histplot(df["price"], bins=30, kde=True)
    plt.title("Price Distribution")
    st.pyplot(plt)
