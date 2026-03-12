# ===============================
# AIRBNB LISTING SUCCESS PREDICTION PROJECT
# ===============================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===============================
# 2. Load Dataset
# ===============================

df = pd.read_csv("listings.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# ===============================
# 3. Data Cleaning
# ===============================

# Remove unnecessary columns if they exist
drop_cols = [
    "id","listing_url","name","description",
    "host_id","host_name","picture_url"
]

df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# Fill missing values
if "reviews_per_month" in df.columns:
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

if "review_scores_rating" in df.columns:
    df["review_scores_rating"] = df["review_scores_rating"].fillna(
        df["review_scores_rating"].mean()
    )

# ===============================
# 4. Feature Engineering
# ===============================

# Convert price column if stored as string
if df["price"].dtype == object:
    df["price"] = df["price"].replace('[\$,]', '', regex=True).astype(float)

# Create target variable
# Listing considered successful if availability is low (means booked frequently)
df["booking_success"] = np.where(df["availability_365"] < 100, 1, 0)

# ===============================
# 5. Select Important Features
# ===============================

features = [
    "price",
    "accommodates",
    "bedrooms",
    "bathrooms",
    "number_of_reviews",
    "review_scores_rating",
    "reviews_per_month",
    "availability_365",
    "room_type",
    "neighbourhood_group"
]

# Keep only existing columns
features = [f for f in features if f in df.columns]

df = df[features + ["booking_success"]]

print("\nSelected Features:")
print(df.head())

# ===============================
# 6. Encode Categorical Variables
# ===============================

le = LabelEncoder()

if "room_type" in df.columns:
    df["room_type"] = le.fit_transform(df["room_type"])

if "neighbourhood_group" in df.columns:
    df["neighbourhood_group"] = le.fit_transform(df["neighbourhood_group"])

# ===============================
# 7. Exploratory Data Analysis
# ===============================

plt.figure(figsize=(8,5))
sns.histplot(df["price"], bins=50)
plt.title("Price Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(x=df["price"], y=df["number_of_reviews"])
plt.title("Price vs Number of Reviews")
plt.show()

plt.figure(figsize=(8,5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# ===============================
# 8. Feature Scaling
# ===============================

X = df.drop("booking_success", axis=1)
y = df["booking_success"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# 9. K-Means Clustering
# ===============================

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df["cluster"] = clusters

plt.figure(figsize=(8,6))
sns.scatterplot(
    x=df["price"],
    y=df["number_of_reviews"],
    hue=df["cluster"],
    palette="viridis"
)

plt.title("Airbnb Listing Clusters")
plt.show()

# ===============================
# 10. Train/Test Split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# ===============================
# 11. Logistic Regression Model
# ===============================

log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train, y_train)

pred_log = log_model.predict(X_test)

log_acc = accuracy_score(y_test, pred_log)

print("\nLogistic Regression Accuracy:", log_acc)

# ===============================
# 12. Random Forest Model
# ===============================

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)

pred_rf = rf_model.predict(X_test)

rf_acc = accuracy_score(y_test, pred_rf)

print("\nRandom Forest Accuracy:", rf_acc)

# ===============================
# 13. Model Evaluation
# ===============================

print("\nClassification Report:")
print(classification_report(y_test, pred_rf))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred_rf))

# ===============================
# 14. Feature Importance
# ===============================

importance = rf_model.feature_importances_

feature_imp = pd.Series(
    importance,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(10,6))
feature_imp.plot(kind="bar")
plt.title("Feature Importance (Random Forest)")
plt.ylabel("Importance Score")
plt.show()

# ===============================
# 15. Cluster Summary
# ===============================

print("\nCluster Summary:")
print(df.groupby("cluster").mean())

# ===============================
# 16. Save Processed Dataset
# ===============================

df.to_csv("processed_airbnb_dataset.csv", index=False)

print("\nProject completed successfully.")
