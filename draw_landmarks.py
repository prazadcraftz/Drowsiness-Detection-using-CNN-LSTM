"""Repo refresh: 2026-04-30"""
import numpy as np
import matplotlib.pyplot as plt
import cv2

def draw_landmarks_on_image(rgb_image, detection_result):
    """Simple visualizer: draw each landmark as a small circle on the image.

    This implementation avoids depending on mediapipe.framework (landmark_pb2)
    and mp.solutions drawing utilities, which can be unavailable in some
    mediapipe distributions. It preserves basic visualization for debugging.
    """
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)
    h, w = annotated_image.shape[:2]

    # Loop through detected faces and draw points
    for face_landmarks in face_landmarks_list:
        for lm in face_landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(annotated_image, (x, y), 1, (0, 255, 0), -1)

    return annotated_image


def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(landmarks, eye_points):
    # Vertical distances
    v1 = euclidean_distance(landmarks[eye_points[0]], landmarks[eye_points[1]])
    v2 = euclidean_distance(landmarks[eye_points[2]], landmarks[eye_points[3]])
    v3 = euclidean_distance(landmarks[eye_points[4]], landmarks[eye_points[5]])
    # Horizontal distance
    h = euclidean_distance(landmarks[eye_points[6]], landmarks[eye_points[7]])
    # EAR formula
    ear = (v1 + v2 + v3) / (3.0 * h)
    return ear

# Function to calculate Mouth Aspect Ratio (MAR)
def calculate_mar(landmarks, mouth_points):
    # Vertical distances
    v1 = euclidean_distance(landmarks[mouth_points[0]], landmarks[mouth_points[1]])
    v2 = euclidean_distance(landmarks[mouth_points[2]], landmarks[mouth_points[3]])
    v3 = euclidean_distance(landmarks[mouth_points[4]], landmarks[mouth_points[5]])
    # Horizontal distance
    h = euclidean_distance(landmarks[mouth_points[6]], landmarks[mouth_points[7]])
    # MAR formula
    mar = (v1 + v2 + v3) / (3.0 * h)
    return mar

# Define 3D model points of the facial landmarks
model_points = np.array([
    (0.0, 0.0, 0.0),     # Nose tip
    (0.0, -330.0, -65.0), # Chin
    (-225.0, 170.0, -135.0), # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0)   # Right mouth corner
], dtype=np.float64)

# Function to calculate head pose
def calculate_head_pose(image, landmarks):
    # Image size
    # print(image)
    size = image.shape
    focal_length = size[1]  # Assume focal length equals image width
    center = (size[1] / 2, size[0] / 2)  # Image center

    # Define the camera matrix
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    # Assume no lens distortion
    dist_coeffs = np.zeros((4, 1))

    # Solve the PnP problem
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, landmarks, camera_matrix, dist_coeffs
    )

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Construct the 3x4 projection matrix by combining rotation matrix and translation vector
    projection_matrix = np.hstack((rotation_matrix, translation_vector))

    # Decompose the projection matrix to get the Euler angles
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)
    
    # Retrieve the angles in degrees
    euler_angles = np.degrees(euler_angles)
    phi = euler_angles[0]  # Left-right rotation
    theta = euler_angles[1]  # Up-down tilt

    return phi, theta

def annotate_points(landmark_li,idx_li, annotated_image, h, w,color):

    for idx in idx_li:
        # Get the x, y coordinates of the landmark
        x = int(landmark_li[idx][0] * w)
        y = int(landmark_li[idx][1] * h)

        # Draw the landmark as a small circle on the frame
        cv2.circle(annotated_image, (x, y), 5, color, -1)
    return annotated_image