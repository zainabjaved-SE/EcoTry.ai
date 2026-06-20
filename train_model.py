# pages/train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
data = pd.read_csv("size_dataset.csv")

# Features
X = data[[
    "gender",
    "shoulder",
    "chest",
    "waist",
    "hip",
    "chest_depth",
    "hip_depth",
    "height",
    "weight"
]]

# Target
y = data["size"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

#  Scaling + MLP in Pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        max_iter=2000,
        random_state=42
    ))
])

# Train
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Save model + encoder
joblib.dump(model, "size_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("Model saved successfully!")