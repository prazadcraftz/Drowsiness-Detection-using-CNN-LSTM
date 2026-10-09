"""Repo refresh: 2026-04-30"""
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import argparse
import os
from pathlib import Path

from config_parser import parse_file
from dataset import get_detector
from draw_landmarks import *
from model import CNN_LSTM

# ============================================================================
# DEBUG FLAG: Set to True to enable detailed output; False for clean output
# ============================================================================
DEBUG = False

def log_debug(msg):
    """Print message only if DEBUG is True."""
    if DEBUG:
        print(f"[DEBUG] {msg}")

def main():
    parser = argparse.ArgumentParser(description='Drowsiness Detection Inference')
    parser.add_argument('--source', type=str, default='0', 
                        help='Video source: 0 for webcam (default), or path to video file')
    parser.add_argument('--save', action='store_true', 
                        help='Save processed video to file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video path (uses config.ini if not specified)')
    args = parser.parse_args()

    # Load configuration
    config_file = parse_file('config.ini')
    log_debug(f"Normalization array: {np.load(config_file['normalize_data_file'])}")

    # Load model
    model = CNN_LSTM(config_file)
    model.load_state_dict(torch.load(config_file['model_path'], weights_only=True))
    model.eval()
    log_debug(f"Model loaded from {config_file['model_path']}")

    # Load normalization data
    min_max_array = np.load(config_file['normalize_data_file'])

    # Setup video source
    if args.source == '0':
        source = 0
        log_debug("Using webcam (source=0)")
    else:
        source = args.source
        if not os.path.exists(source):
            print(f"Error: Video file '{source}' not found")
            return
        log_debug(f"Using video file: {source}")

    # Open video capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{source}'")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Default FPS if not available

    # Setup video writer if save is requested
    video_writer = None
    if args.save:
        output_path = args.output if args.output else config_file.get('inference_video_out', 'outputs/output.mp4')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        print(f"Saving output to: {output_path}")

    # Load face detector
    detector = get_detector(config_file['mediapipe_model_path'])

    # Feature buffers
    frame_buffer = []
    r_ear_li = []
    l_ear_li = []
    mar_li = []
    phi_li = []
    theta_li = []

    # Class mapping (assuming 3 classes: Normal, Yawning, Eye Closed)
    class_names = ['ATTENTIVE', 'DROWSY', 'DROWSY']

    frame_count = 0

    print("Running inference... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            log_debug("End of video or camera disconnected")
            break

        frame_count += 1

        # Detect face landmarks
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = detector.detect(image)

        try:
            landmarks = np.array([(face_landmarks.x, face_landmarks.y) 
                                 for face_landmarks in detection_result.face_landmarks[0]])
        except (IndexError, AttributeError):
            if DEBUG:
                print(f"[DEBUG] Frame {frame_count}: No face detected")
            cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 0, 255), 2)
            if video_writer:
                video_writer.write(frame)
            cv2.imshow('Drowsiness Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Calculate facial features
        r_ear = calculate_ear(landmarks, [160, 144, 159, 145, 158, 153, 33, 133])
        l_ear = calculate_ear(landmarks, [385, 380, 386, 374, 387, 373, 362, 263])
        mar = calculate_mar(landmarks, [81, 178, 13, 14, 311, 402, 78, 308])
        phi, theta = calculate_head_pose(frame, np.array([landmarks[i] for i in [10, 33, 263, 152, 61, 291]]))

        # Normalize features
        r_ear_norm = (r_ear - min_max_array[0][0]) / (min_max_array[0][1] - min_max_array[0][0])
        l_ear_norm = (l_ear - min_max_array[1][0]) / (min_max_array[1][1] - min_max_array[1][0])
        mar_norm = (mar - min_max_array[2][0]) / (min_max_array[2][1] - min_max_array[2][0])
        phi_norm = (phi[0] - min_max_array[3][0]) / (min_max_array[3][1] - min_max_array[3][0])
        theta_norm = (theta[0] - min_max_array[4][0]) / (min_max_array[4][1] - min_max_array[4][0])

        r_ear_li.append(r_ear_norm)
        l_ear_li.append(l_ear_norm)
        mar_li.append(mar_norm)
        phi_li.append(phi_norm)
        theta_li.append(theta_norm)

        all_features = np.vstack([r_ear_li, l_ear_li, mar_li, phi_li, theta_li])
        frame_buffer.append(all_features[:, len(r_ear_li) - 1])

        # Keep buffer to 16 frames
        if len(frame_buffer) > 16:
            frame_buffer.pop(0)

        # Run inference when buffer is full
        prediction_text = "Loading..."
        confidence = 0
        
        if len(frame_buffer) == 16:
            fr = np.array(frame_buffer)
            fr = np.transpose(fr, (1, 0))
            log_debug(f"Frame {frame_count}: Input shape {fr.shape}")
            
            input_data = torch.tensor(fr, dtype=torch.float32).unsqueeze(axis=0)
            log_debug(f"Frame {frame_count}: Tensor shape {input_data.shape}")

            # Make prediction
            with torch.no_grad():
                prediction = model(input_data)
            
            # Get predicted class and confidence
            softmax = nn.Softmax(dim=1)(prediction)
            predicted_class = torch.argmax(softmax, 1).item()
            confidence = softmax[0, predicted_class].item()
            
            prediction_text = f"{class_names[predicted_class]} ({confidence*100:.1f}%)"
            log_debug(f"Frame {frame_count}: {prediction_text}")

        # Overlay information on frame
        h, w = frame.shape[:2]
        
        # Top-left label with status and confidence
        cv2.putText(frame, "Status:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 2)
        color = (0, 255, 0) if "ATTENTIVE" in prediction_text else (0, 0, 255)
        cv2.putText(frame, prediction_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, color, 2)

        # Optional: Show facial metrics (only if buffer is full for cleaner display)
        if len(frame_buffer) == 16:
            metrics_text = f"R-EAR: {r_ear:.2f} L-EAR: {l_ear:.2f} MAR: {mar:.2f}"
            cv2.putText(frame, metrics_text, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (200, 200, 200), 1)

        # Display frame
        cv2.imshow('Drowsiness Detection', frame)

        # Save frame if requested
        if video_writer:
            video_writer.write(frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    # Cleanup
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    print("Inference complete.")

if __name__ == '__main__':
    main()
