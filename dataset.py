"""Repo refresh: 2026-04-30"""
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import cv2
from draw_landmarks import *
import numpy as np
from tqdm import tqdm

def get_detector(model_path):

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=True,
                                        output_facial_transformation_matrixes=True,
                                        num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    return detector

def create_windows(all_features, label, class_idx, window_size=16, overlap=8):
    num_features, num_frames = all_features.shape
    step = window_size - overlap
    windows = []
    labels = []
    # Sliding window approach
    for start in range(0, num_frames, step):
        end = start + window_size

        if end <= num_frames:
            # If the window is fully within the number of frames, take it as is
            window = all_features[:, start:end]
        else:
            # If we reach the end and the window is incomplete, pad by duplicating the last frame
            window = all_features[:, start:num_frames]
            last_frame = all_features[:, -1:]  # Take the last frame

            # Duplicate the last frame to fill the remaining space
            padding = np.repeat(last_frame, window_size - window.shape[1], axis=1)
            window = np.hstack([window, padding])

        windows.append(window)
        labels.append(class_idx[label])

    return windows,labels

def preprocess_data(config):
    
    detector = get_detector(config['mediapipe_model_path'])
    directory = config['train_dataset_path']
    classes_to_capture = os.listdir(directory)
    # classes_to_capture = ['Normal','Yawning']
    idx = np.arange(len(classes_to_capture))
    class_idx = dict(zip(classes_to_capture,idx))

    all_windows = []
    all_labels = []

    r_ear_minmax = [0,0]
    l_ear_minmax = [0,0]
    mar_minmax = [0,0]
    phi_minmax = [0,0]
    theta_minmax = [0,0]


    for fol in tqdm(classes_to_capture):
        # print(fol)
        for videos in tqdm(os.listdir(os.path.join(directory, fol))):
            cap = cv2.VideoCapture(os.path.join(directory,fol,videos))
            if (cap.isOpened()== False): 
                print("Error opening video stream or file")
            # Read until video is completed
            # cnt=0
            r_ear_li = []
            l_ear_li = []
            mar_li = []
            phi_li = []
            theta_li = []
            f = 0
            while(cap.isOpened()):
                # Capture frame-by-frame
                # print(cnt)
                ret, frame = cap.read()
                # print(directory,fol,videos)
                # f = frame
                if ret == True:
                    # STEP 3: Load the input image.
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                    # STEP 4: Detect face landmarks from the input image.
                    detection_result = detector.detect(image)
                    # STEP 5: Process the detection result. In this case, visualize it.
                    #annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
                    # cv2.imshow('img',cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                    try:
                        landmarks = np.array([(face_landmarks.x, face_landmarks.y) for face_landmarks in detection_result.face_landmarks[0]])
                    except:
                        print('No landmark',fol,videos)
                        # cv2.imshow('frame',frame)
                        # cv2.waitKey(0)
                        continue
                    r_ear = calculate_ear(landmarks, [160, 144, 159, 145, 158, 153, 33, 133])
                    # Left Eye Aspect Ratio (L-EAR) calculation
                    l_ear = calculate_ear(landmarks, [385, 380, 386, 374, 387, 373, 362, 263])
                    # Mouth Aspect Ratio (MAR) calculation
                    mar = calculate_mar(landmarks, [81, 178, 13, 14, 311, 402, 78, 308])
                    # Calculate head pose
                    phi, theta = calculate_head_pose(frame, np.array([landmarks[i] for i in [10,33,263,152,61,291]]))
                    r_ear_li.append(r_ear)
                    l_ear_li.append(l_ear)
                    mar_li.append(mar)
                    phi_li.append(phi[0])
                    theta_li.append(theta[0])
                else:
                    break
                # cnt+=1
            cap.release()

            r_ear_minmax[0] = min(min(r_ear_li),r_ear_minmax[0])
            l_ear_minmax[0] = min(min(l_ear_li),l_ear_minmax[0])
            mar_minmax[0] = min(min(mar_li),mar_minmax[0])
            phi_minmax[0] = min(min(phi_li),phi_minmax[0])
            theta_minmax[0] = min(min(theta_li),theta_minmax[0])

            r_ear_minmax[1] = max(max(r_ear_li),r_ear_minmax[1])
            l_ear_minmax[1] = max(max(l_ear_li),l_ear_minmax[1])
            mar_minmax[1] = max(max(mar_li),mar_minmax[1])
            phi_minmax[1] = max(max(phi_li),phi_minmax[1])
            theta_minmax[1] = max(max(theta_li),theta_minmax[1])

            all_features = np.vstack([r_ear_li, l_ear_li, mar_li, phi_li, theta_li])
            windows, labels = create_windows(all_features, fol,class_idx ,window_size=config['window_size'], overlap=config['overlap'])
            all_windows.extend(windows)
            all_labels.extend(labels)

    windows_arr = np.array(all_windows)
    labels_array = np.array(all_labels)

    windows_arr[:,0,:] = (windows_arr[:,0,:] - r_ear_minmax[0]) / (r_ear_minmax[1] - r_ear_minmax[0])
    windows_arr[:,1,:] = (windows_arr[:,1,:] - l_ear_minmax[0]) / (l_ear_minmax[1] - l_ear_minmax[0])
    windows_arr[:,2,:] = (windows_arr[:,2,:] - mar_minmax[0]) / (mar_minmax[1] - mar_minmax[0])
    windows_arr[:,3,:] = (windows_arr[:,3,:] - phi_minmax[0]) / (phi_minmax[1] - phi_minmax[0])
    windows_arr[:,4,:] = (windows_arr[:,4,:] - theta_minmax[0]) / (theta_minmax[1] - theta_minmax[0])

    norm = np.vstack([r_ear_minmax,l_ear_minmax,mar_minmax,phi_minmax,theta_minmax])
    np.save(config['normalize_data_file'],norm)
    np.save(config['preprocessed_windows_data'],windows_arr)
    np.save(config['preprocessed_labels_data'],labels_array)

    return class_idx,windows_arr,labels_array


