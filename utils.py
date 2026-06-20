# pages/utils.py
import mediapipe as mp
import numpy as np
import cv2

mp_pose = mp.solutions.pose

def get_body_measurements(image, height_cm):

    pose = mp_pose.Pose(static_image_mode=True)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks.landmark

    # Basic landmark points
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_hip = landmarks[23]
    right_hip = landmarks[24]

    # Shoulder width (pixel distance)
    shoulder_px = abs(left_shoulder.x - right_shoulder.x)

    # Convert to approximate cm using height ratio
    shoulder_cm = shoulder_px * height_cm * 0.25

    # Generate realistic proportional values
    chest = shoulder_cm * 2.1
    waist = shoulder_cm * 1.8
    hip = shoulder_cm * 2.2
    chest_depth = shoulder_cm * 0.6
    hip_depth = shoulder_cm * 0.65

    return {
        "shoulder": round(shoulder_cm, 2),
        "chest": round(chest, 2),
        "waist": round(waist, 2),
        "hip": round(hip, 2),
        "chest_depth": round(chest_depth, 2),
        "hip_depth": round(hip_depth, 2),
    }