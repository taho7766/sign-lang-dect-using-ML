import cv2 as cv
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles

def process_mp(image, model):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(frame, results):
    mp_drawing.draw_landmarks(
        frame,
        results.face_landmarks,
        mp_holistic.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_style
        .get_default_face_mesh_tesselation_style()
    )
    mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_style
        .get_default_pose_landmarks_style()
    )
    mp_drawing.draw_landmarks(
        frame,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_style
        .get_default_hand_landmarks_style()
    )
    mp_drawing.draw_landmarks(
        frame,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_style
        .get_default_hand_landmarks_style()
    )
    
def extract_keypoints(results):
    if results.right_hand_landmarks:
        right_hand_landmark_points = np.array([[res.x, res.y, res.z, res.visibility] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        right_hand_landmark_points = np.zeros(21 * 4)

    if results.left_hand_landmarks:
        left_hand_landmark_points = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        left_hand_landmark_points = np.zeros(21 * 3)
        
    if results.pose_landmarks:
        pose_landmark_points = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose_landmark_points = np.zeros(33 * 3)

    if results.face_landmarks:
        face_landmark_points = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()
    else:
        face_landmark_points = np.zeros(468 * 3)
    return np.concatenate([right_hand_landmark_points, left_hand_landmark_points, pose_landmark_points, face_landmark_points])