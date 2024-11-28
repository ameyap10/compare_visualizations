#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load all datasets
datasets = {
    "Biased Dataset": pd.read_csv("Complex_Biased_Last_Mile_Delivery_Dataset.csv"),
    "Fair Dataset (CMM)": pd.read_csv("Fair_Causal_Mitigated_Dataset.csv"),
    "Reweighted Dataset": pd.read_csv("Reweighted_Last_Mile_Delivery_Dataset.csv"),
    "SMOTE Dataset": pd.read_csv("SMOTE_Last_Mile_Delivery_Dataset.csv"),
}

# Preprocess datasets (mapping encoded values back to categories for Reweighted and SMOTE datasets)
def preprocess_encoded_data(df):
    mapping = {
        "route_type": {0: "Urban", 1: "Rural"},
        "customer_priority": {0: "Low", 1: "Medium", 2: "High"},
        "customer_type": {0: "Individual", 1: "Corporate"},
        "driver_experience": {0: "Novice", 1: "Intermediate", 2: "Expert"},
        "vehicle_type": {0: "Bike", 1: "Van", 2: "Truck"},
        "weather_conditions": {0: "Clear", 1: "Rainy", 2: "Snowy", 3: "Foggy"},
        "traffic_level": {0: "Low", 1: "Medium", 2: "High"},
        "day_of_week": {0: "Monday", 1: "Tuesday", 2: "Wednesday",
                        3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"},
        "delivery_window": {0: "Morning", 1: "Afternoon", 2: "Evening"}
    }
    for col, mapping_dict in mapping.items():
        if col in df.columns:
            df[col] = df[col].map(mapping_dict)
    return df

datasets["Reweighted Dataset"] = preprocess_encoded_data(datasets["Reweighted Dataset"])
datasets["SMOTE Dataset"] = preprocess_encoded_data(datasets["SMOTE Dataset"])

# Interactive Sidebar
st.sidebar.title("Interactive Comparison Website")
selected_dataset = st.sidebar.selectbox("Select a Dataset", datasets.keys())
selected_visualization = st.sidebar.selectbox(
    "Choose a Visualization",
    [
        "Average Delivery Times by Route Type",
        "Delivery Time Distribution by Customer Priority",
        "Success Rate by Route Type",
        "Correlation Heatmap",
    ]
)

# Display Title
st.title("Dataset Comparison")
st.write(f"### {selected_visualization} for {selected_dataset}")

# Get selected dataset
data = datasets[selected_dataset]

# Visualization Logic
if selected_visualization == "Average Delivery Times by Route Type":
    plt.figure(figsize=(10, 6))
    sns.barplot(data=data, x="route_type", y="actual_time", ci=None, palette="viridis")
    plt.title("Average Delivery Times by Route Type")
    plt.xlabel("Route Type")
    plt.ylabel("Average Delivery Time")
    st.pyplot()

elif selected_visualization == "Delivery Time Distribution by Customer Priority":
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data, x="customer_priority", y="actual_time", hue="route_type", palette="muted")
    plt.title("Delivery Time Distribution by Customer Priority and Route Type")
    plt.xlabel("Customer Priority")
    plt.ylabel("Delivery Time (minutes)")
    st.pyplot()

elif selected_visualization == "Success Rate by Route Type":
    plt.figure(figsize=(10, 6))
    sns.barplot(data=data, x="route_type", y="is_cutoff", ci=None, palette="coolwarm")
    plt.title("Success Rate by Route Type")
    plt.xlabel("Route Type")
    plt.ylabel("Success Rate")
    st.pyplot()

elif selected_visualization == "Correlation Heatmap":
    plt.figure(figsize=(10, 8))
    corr = data.select_dtypes(include=['number']).corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Numeric Variables")
    st.pyplot()

# Additional Features (Optional):
# Add descriptive statistics or dataset previews
if st.sidebar.checkbox("Show Dataset Preview"):
    st.write(data.head())


# In[ ]:




