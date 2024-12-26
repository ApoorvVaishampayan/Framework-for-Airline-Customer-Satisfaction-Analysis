#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 14:10:48 2024

@author: apoorvvaishampayan

#Command to run the code through Terminal :-  streamlit run ~/Downloads/MIS695script.py

"""
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import plotly.express as px


st.set_page_config(
    page_title="Airline Satisfaction Analysis",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("✈️ Airline Passenger Satisfaction Analysis")
st.markdown(
    """
    **Welcome to the Airline Passenger Satisfaction Analysis App!**  
    Explore how various factors like flight distance, delays, and onboard services impact customer satisfaction.
    """
)

st.header("📂 Upload Your Dataset")
uploaded_file = st.file_uploader("Upload an Excel file:", type=["xlsx"])

if uploaded_file:
    data = pd.read_excel(uploaded_file)
    st.subheader("🔍 Dataset Preview")
    st.write(data.head())

    st.markdown("---")
    st.header("🛠️ Data Preprocessing")
    st.write("### Encoding Categorical Variables & Handling missing values")
    data['Arrival Delay in Minutes'].fillna(0, inplace=True)
    data['Departure Delay in Minutes'].fillna(0, inplace=True)
    data = data.drop(columns=['id'])  # Drop irrelevant columns
    categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
    data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    st.markdown("---")
    st.header("📊 Exploratory Data Analysis (EDA)")
    st.subheader("Pie Charts for Categorical Variables")
    categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
    for i in range(0, len(categorical_columns), 2):  # Iterate two at a time
        cols = st.columns(2)
        for j in range(2):  # Two charts per row
            if i + j < len(categorical_columns):
                col = categorical_columns[i + j]
                with cols[j]:  # Assign each chart to a column
                    st.write(f"### {col}")
                    fig, ax = plt.subplots(figsize=(5, 5))  # Adjusted size
                    counts = data[col].value_counts()
                    ax.pie(
                        counts, labels=counts.index, autopct='%1.1f%%', 
                        colors=sns.color_palette('pastel'), startangle=90
                    )
                    ax.set_title(f'Distribution of {col}')
                    st.pyplot(fig)

    st.subheader("Bar Plots for Satisfaction-Related Variables")
    kpi_columns = [
        "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking", 
        "Gate location", "Food and drink", "Online boarding", "Seat comfort", 
        "Inflight entertainment", "On-board service", "Leg room service", 
        "Baggage handling", "Checkin service", "Inflight service", "Cleanliness"
    ]

    for i in range(0, len(kpi_columns), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(kpi_columns):
                col = kpi_columns[i + j]
                with cols[j]:
                    st.write(f"### {col}")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    order = sorted(data[col].unique())
                    sns.countplot(x=col, data=data, palette='viridis', order=order, ax=ax)
                    ax.set_title(f'Distribution of {col}')
                    ax.set_xlabel(f'{col} Ratings')
                    ax.set_ylabel('Frequency')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

    st.markdown("---")
    st.header("📊 Advanced Exploratory Data Analysis (EDA)")
    if all(col in data.columns for col in ["satisfaction", "Gender", "Customer Type", "Type of Travel", "Class"]):
        st.subheader("Sunburst Chart: Satisfaction vs Gender")
        sunburst_fig_gender = px.sunburst(
            data,
            path=["satisfaction", "Gender"],
            color="satisfaction",
            color_discrete_map={"satisfied": "green", "neutral or dissatisfied": "red"},
            title="Satisfaction vs Gender",
            width=700,
            height=700
        )
        st.plotly_chart(sunburst_fig_gender)

        st.subheader("Sunburst Chart: Satisfaction vs Customer Type")
        sunburst_fig_customer_type = px.sunburst(
            data,
            path=["satisfaction", "Customer Type"],
            color="satisfaction",
            color_discrete_map={"satisfied": "green", "neutral or dissatisfied": "red"},
            title="Satisfaction vs Customer Type",
            width=700,
            height=700
        )
        st.plotly_chart(sunburst_fig_customer_type)

        st.subheader("Sunburst Chart: Satisfaction vs Type of Travel")
        sunburst_fig_travel_type = px.sunburst(
            data,
            path=["satisfaction", "Type of Travel"],
            color="satisfaction",
            color_discrete_map={"satisfied": "green", "neutral or dissatisfied": "red"},
            title="Satisfaction vs Type of Travel",
            width=700,
            height=700
        )
        st.plotly_chart(sunburst_fig_travel_type)

        st.subheader("Sunburst Chart: Satisfaction vs Class")
        sunburst_fig_class = px.sunburst(
            data,
            path=["satisfaction", "Class"],
            color="satisfaction",
            color_discrete_map={"satisfied": "green", "neutral or dissatisfied": "red"},
            title="Satisfaction vs Class",
            width=700,
            height=700
        )
        st.plotly_chart(sunburst_fig_class)
        
    
    kpi_columns = [
        "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
        "Gate location", "Food and drink", "Online boarding", "Seat comfort",
        "Inflight entertainment", "On-board service", "Leg room service",
        "Baggage handling", "Checkin service", "Inflight service", "Cleanliness"
    ]

    if all(col in data.columns for col in kpi_columns):
        for i in range(0, len(kpi_columns), 2):  
            cols = st.columns(2) 
            
            if i < len(kpi_columns):
                with cols[0]:
                    kpi = kpi_columns[i]
                    fig, ax = plt.subplots(figsize=(5, 3))
                    sns.countplot(data=data, x=kpi, hue="satisfaction", palette="coolwarm", ax=ax)
                    ax.set_title(f"{kpi}")
                    ax.set_xlabel("Ratings")
                    ax.set_ylabel("Count")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

            if i + 1 < len(kpi_columns):
                with cols[1]:
                    kpi = kpi_columns[i + 1]
                    fig, ax = plt.subplots(figsize=(5, 3))
                    sns.countplot(data=data, x=kpi, hue="satisfaction", palette="coolwarm", ax=ax)
                    ax.set_title(f"{kpi}")
                    ax.set_xlabel("Ratings")
                    ax.set_ylabel("Count")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

    st.markdown("---")
    st.header("📊 KPI Analysis: Key Satisfaction Metrics")
    
    satisfaction_columns = [
           "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking", 
           "Gate location", "Food and drink", "Online boarding", "Seat comfort", 
           "Inflight entertainment", "On-board service", "Leg room service", 
           "Baggage handling", "Checkin service", "Inflight service", "Cleanliness"
       ]
    kpi_data = data[satisfaction_columns]
    kpi_averages = kpi_data.mean().sort_values(ascending=False)
    leading_kpi = kpi_averages.idxmax()
    lagging_kpi = kpi_averages.idxmin()
    st.write(f"**Leading KPI**: {leading_kpi} with an average score of **{kpi_averages.max():.2f}**")
    st.write(f"**Lagging KPI**: {lagging_kpi} with an average score of **{kpi_averages.min():.2f}**")

    st.write("### KPI Average Ratings")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=kpi_averages.values, y=kpi_averages.index, palette="viridis", ax=ax)
    ax.set_title("Average Ratings for Satisfaction KPIs")
    ax.set_xlabel("Average Score (0–5)")
    st.pyplot(fig)
    
    kpi_icons = {
        "Inflight wifi service": "📡",
        "Departure/Arrival time convenient": "⏰",
        "Ease of Online booking": "💻",
        "Gate location": "📍",
        "Food and drink": "🍴",
        "Online boarding": "📱",
        "Seat comfort": "💺",
        "Inflight entertainment": "🎥",
        "On-board service": "🛎️",
        "Leg room service": "🦵",
        "Baggage handling": "🧳",
        "Checkin service": "🛂",
        "Inflight service": "✈️",
        "Cleanliness": "🧼",
    }

    kpi_table = pd.DataFrame({
        "KPI": [f"{kpi_icons[kpi]} {kpi}" for kpi in kpi_averages.index],
        "Average Score": kpi_averages.values,
        "Ranking (Attention Needed)": range(1, len(kpi_averages) + 1)
    })

    st.write("### KPI Ranking Table")
    st.write(kpi_table.style.format({"Average Score": "{:.2f}"}))


    st.markdown("---")
    st.header("📊 Breakdown Analysis by Demographics")
    
    breakdown_columns = ["Gender", "Customer Type", "Type of Travel", "Class"]
    selected_column = st.selectbox("Select a demographic for breakdown:", breakdown_columns)
    breakdown_column = selected_column
    grouped_kpis = data.groupby(breakdown_column)[satisfaction_columns].mean()
    st.write(f"### Average KPI Ratings by {breakdown_column}")
    st.write(grouped_kpis)
    data_cut_icons = {
        "Gender": "👥",
        "Customer Type": "👤",
        "Type of Travel": "✈️",
        "Class": "🏷️"
    }
    
    best_kpis = grouped_kpis.idxmax(axis=1)  
    worst_kpis = grouped_kpis.idxmin(axis=1)  
    best_ratings = grouped_kpis.max(axis=1)  
    worst_ratings = grouped_kpis.min(axis=1)  
    
    kpi_table_with_cuts = pd.DataFrame({
        "Data Cuts": [
            f"{data_cut_icons[breakdown_column]} {value}" for value in grouped_kpis.index
        ],
        "Best KPI": [
            f"{kpi_icons[best_kpis[value]]} {best_kpis[value]}" for value in grouped_kpis.index
        ],
        "Best Rating": [best_ratings[value] for value in grouped_kpis.index],
        "Worst KPI": [
            f"{kpi_icons[worst_kpis[value]]} {worst_kpis[value]}" for value in grouped_kpis.index
        ],
        "Worst Rating": [worst_ratings[value] for value in grouped_kpis.index]
    })
    st.write("### Data Cuts with Best and Worst Performing KPIs")
    st.dataframe(
        kpi_table_with_cuts.style.format(
            {
                "Best Rating": "{:.2f}",
                "Worst Rating": "{:.2f}"
            }
        )
    )

    st.markdown("---")
    st.header("🧠 Model Training and Evaluation")
    X = data_encoded.drop(columns=['satisfaction_satisfied'])
    y = data_encoded['satisfaction_satisfied']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.subheader("📌 Naive Bayes Model")
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    nb_predictions = nb_model.predict(X_test)
    nb_accuracy = accuracy_score(y_test, nb_predictions)
    st.write(f"**Accuracy:** {nb_accuracy:.2%}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, nb_predictions))
    st.subheader("📌 Decision Tree Model")
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    dt_predictions = dt_model.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_predictions)
    st.write(f"**Accuracy:** {dt_accuracy:.2%}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, dt_predictions))

    st.write("### Confusion Matrices")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(confusion_matrix(y_test, nb_predictions), annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title("Naive Bayes Confusion Matrix")
    sns.heatmap(confusion_matrix(y_test, dt_predictions), annot=True, fmt="d", cmap="Greens", ax=axes[1])
    axes[1].set_title("Decision Tree Confusion Matrix")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.header("💡 Recommendations")
    st.write("""
    - Focus on improving inflight services like WiFi, entertainment, and legroom, as these strongly impact satisfaction.
    - Address delays proactively by improving scheduling and communication.
    - Enhance seat comfort in economy class to improve satisfaction for the largest customer base.
    """)

else:
    st.warning("⚠️ Please upload a dataset to begin!")


