# codealpha_sentiment-analysis1


# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from textblob import TextBlob

# Set visual style for plots
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

print("=" * 50)
print("TASK 3: DATA VISUALIZATION")
print("=" * 50)

# Load sample dataset (Titanic dataset) for Data Visualization
df_viz = sns.load_dataset("titanic")
print(
    f"Loaded Titanic Dataset for Visualization. Shape: {df_viz.shape[0]} rows, {df_viz.shape[1]} columns\n"
)

# 1. Bar Chart / Count Plot: Survival by Class and Gender
plt.figure(figsize=(9, 5))
sns.countplot(data=df_viz, x="class", hue="survived", palette="Set2")
plt.title(
    "Survival Count by Passenger Class and Gender",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Passenger Class", fontsize=12)
plt.ylabel("Passenger Count", fontsize=12)
plt.legend(title="Survived", labels=["No", "Yes"])
plt.tight_layout()
plt.show()

# 2. Histogram / Distribution Plot: Age Distribution by Survival
plt.figure(figsize=(9, 5))
sns.histplot(
    data=df_viz,
    x="age",
    hue="survived",
    multiple="stack",
        bins=30,
    palette="viridis",
)
plt.title(
    "Age Distribution Stacked by Survival Status",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Age", fontsize=12)
plt.ylabel("Passenger Count", fontsize=12)
plt.tight_layout()
plt.show()

# 3. Correlation Heatmap for Numerical Variables
plt.figure(figsize=(8, 6))
numeric_df = df_viz.select_dtypes(include=[np.number])
sns.heatmap(
    numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
)
plt.title(
    "Correlation Heatmap of Numerical Features", fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.show()

print("\n" + "=" * 50)
print("TASK 4: SENTIMENT ANALYSIS")
print("=" * 50)

# Sample text data (simulating customer reviews / social media comments)
reviews_data = {
    "review_text": [
        "I absolutely love this product! It exceeded all my expectations.",
        "Terrible experience. The item broke after just two days of use.",
        "The service was okay, nothing special to write home about.",
        "Fantastic quality and fast shipping! Highly recommended to everyone.",
        "Worst customer support ever. Completely unhelpful and rude.",
        "It's an average product for a decent price. Gets the job done.",
    ]
}

# Create a DataFrame for Sentiment Analysis
df_sentiment = pd.DataFrame(reviews_data)


# Function to get polarity sentiment and classify
def analyze_sentiment(text):
  analysis = TextBlob(text)
  polarity = analysis.sentiment.polarity
  if polarity > 0.1:
    return "Positive"
  elif polarity < -0.1:
    return "Negative"
  else:
    return "Neutral"


# Apply sentiment analysis to dataset
df_sentiment["Polarity"] = df_sentiment["review_text"].apply(
    lambda x: TextBlob(x).sentiment.polarity
)
df_sentiment["Sentiment"] = df_sentiment["review_text"].apply(analyze_sentiment)

# Display the analyzed dataframe
print("Sentiment Analysis Results:")
print(df_sentiment[["review_text", "Polarity", "Sentiment"]])
print("\n")

# Summary counts of sentiments
sentiment_counts = df_sentiment["Sentiment"].value_counts()
print("Sentiment Summary Distribution:")
print(sentiment_counts)

# Visualization for Sentiment Analysis
plt.figure(figsize=(7, 4))
sns.countplot(
    data=df_sentiment, x="Sentiment", order=["Positive", "Neutral", "Negative"], palette="pastel"
)
plt.title(
    "Distribution of Sentiments in Sample Reviews",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Sentiment Class", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.tight_layout()
plt.show()

print("\n--- Task 3 & Task 4 Executed Successfully! ---")
