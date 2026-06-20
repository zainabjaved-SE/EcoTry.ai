import joblib
import numpy as np

model = joblib.load("size_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

def recommend_size(measurements, height, weight, gender):

    features = np.array([[
        gender,
        measurements["shoulder"],
        measurements["chest"],
        measurements["waist"],
        measurements["hip"],
        measurements["chest_depth"],
        measurements["hip_depth"],
        height,
        weight
    ]])

    prediction = model.predict(features)
    predicted_size = label_encoder.inverse_transform(prediction)

    return predicted_size[0]