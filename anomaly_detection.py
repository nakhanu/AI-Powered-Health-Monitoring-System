import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("simulated_health_data.csv")

# Extract features and true labels
X = df[["heart_rate", "spo2"]]
true_labels = df["label"].map({"Normal": 1, "Anomaly": -1})

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Isolation Forest
model = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
preds = model.fit_predict(X_scaled)

# Evaluate
precision = precision_score(true_labels, preds, pos_label=-1)
recall = recall_score(true_labels, preds, pos_label=-1)
f1 = f1_score(true_labels, preds, pos_label=-1)

print("\nModel Evaluation")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")

# Save predictions
df["predicted_label"] = ["Anomaly" if p == -1 else "Normal" for p in preds]
df.to_csv("anomaly_detection_results.csv", index=False)
print("\nResults saved as anomaly_detection_results.csv")

import matplotlib.pyplot as plt

# Map labels to colors
color_map = {'Normal': 'green', 'Anomaly': 'red'}
colors = df['predicted_label'].map(color_map)

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['heart_rate'], df['spo2'], c=colors, alpha=0.7, edgecolor='k')
plt.title("Anomaly Detection in Health Data")
plt.xlabel("Heart Rate (bpm)")
plt.ylabel("SpO₂ (%)")
plt.grid(True)
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', label='Normal', markerfacecolor='green', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Anomaly', markerfacecolor='red', markersize=10)
])
plt.tight_layout()
plt.savefig("anomaly_scatter_plot.png")
plt.show()

