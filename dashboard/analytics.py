import pandas as pd
import matplotlib.pyplot as plt
import seaborn as plt_sns
import numpy as np

def generate_dummy_data():
    """Generates a dummy dataset for analytics purposes."""
    np.random.seed(42)
    diseases = ["Healthy", "Apple Scab", "Apple Black Rot", "Apple Cedar Rust", "Corn Common Rust", "Tomato Blight"]
    records = []
    
    # Generate 100 historical records
    for _ in range(200):
        records.append({
            "Disease": np.random.choice(diseases, p=[0.3, 0.15, 0.1, 0.1, 0.15, 0.2]),
            "Month": np.random.choice(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "August"]),
            "Crop_Type": np.random.choice(["Apple", "Corn", "Tomato"], p=[0.5, 0.25, 0.25]),
            "Confidence_Score": np.random.uniform(0.7, 0.99)
        })
        
    return pd.DataFrame(records)

def plot_disease_distribution(df):
    """Returns a Matplotlib figure showing disease distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df['Disease'].value_counts()
    plt_sns.barplot(x=counts.values, y=counts.index, ax=ax, palette="viridis")
    ax.set_title("Disease Distribution in Historical Scans")
    ax.set_xlabel("Count of Scans")
    ax.set_ylabel("Disease Type")
    plt.tight_layout()
    return fig

def plot_disease_trends(df):
    """Returns a Matplotlib figure showing occurrence trends over months."""
    fig, ax = plt.subplots(figsize=(10, 5))
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "August"]
    counts = df.groupby(['Month', 'Disease']).size().unstack().fillna(0)
    counts = counts.reindex(month_order)
    
    counts.plot(kind='line', marker='o', ax=ax, cmap="tab10")
    ax.set_title("Disease Occurrence Trends Over Time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Detected Cases")
    ax.legend(title='Disease', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig

def plot_crop_statistics(df):
    """Returns a Matplotlib figure summarizing crop statistics."""
    fig, ax = plt.subplots(figsize=(6, 6))
    crop_counts = df['Crop_Type'].value_counts()
    ax.pie(crop_counts, labels=crop_counts.index, autopct='%1.1f%%', colors=plt_sns.color_palette("Set2"), startangle=90)
    ax.set_title("Crop Type Analysis")
    return fig
