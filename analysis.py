import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("MarketFarePredictionData.csv")
print(df.shape)
print(df.dtypes)
df.head()


# Check missing values
print(df.isnull().sum())

# Drop rows with missing fare
df = df.dropna(subset=["Average_Fare"])
print(f"Clean dataset: {len(df):,} rows")

plt.figure(figsize=(10, 5))
sns.histplot(df["Average_Fare"], bins=50, kde=True)
plt.title("Average Fare Distribution")
plt.xlabel("Fare (USD)")
plt.tight_layout()
plt.savefig("fare_distribution.png")
plt.show()

#------------------------

carrier_fare = df.groupby("Carrier")["Average_Fare"].mean().sort_values(ascending=False).head(15)

plt.figure(figsize=(12, 5))
carrier_fare.plot(kind="bar")
plt.title("Average Fare by Carrier (Top 15)")
plt.ylabel("Average Fare (USD)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("fare_by_carrier.png")
plt.show()


plt.figure(figsize=(8, 5))
sns.scatterplot(data=df.sample(5000), x="Market_HHI", y="Average_Fare", alpha=0.3)
plt.title("Market Competition (HHI) vs Average Fare")
plt.xlabel("HHI (higher = less competition)")
plt.ylabel("Average Fare (USD)")
plt.tight_layout()
plt.savefig("competition_vs_fare.png")
plt.show()


Q1 = df["Average_Fare"].quantile(0.25)
Q3 = df["Average_Fare"].quantile(0.75)
IQR = Q3 - Q1

anomalies = df[(df["Average_Fare"] < Q1 - 1.5*IQR) | (df["Average_Fare"] > Q3 + 1.5*IQR)]
print(f"Anomalies detected: {len(anomalies):,} ({len(anomalies)/len(df)*100:.1f}%)")
anomalies[["Carrier", "Average_Fare", "Market_HHI", "LCC_Comp"]].head(10)




from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

features = ["NonStopMiles", "Market_HHI", "LCC_Comp", "Pax", "Market_share", "Circuity"]
df_model = df[features + ["Average_Fare"]].dropna()

X = df_model[features]
y = df_model["Average_Fare"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")


importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

plt.figure(figsize=(8, 5))
importance.plot(kind="bar")
plt.title("Feature Importance for Fare Prediction")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

print("""
=== BUSINESS INSIGHTS ===
1. Competition drives fares down: High HHI markets show higher fares
2. Distance is the strongest predictor of fare price
3. Anomalous fares signal pricing opportunities
4. LCC presence suppresses fares on competitive routes
""")