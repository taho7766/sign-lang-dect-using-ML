import numpy as np
import cv2 as cv
import mediapipe as mp
import cProfile

mp_drawing = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles
mp_holistics = mp.solutions.holistic

def process_mp(frame, holistic):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = holistic.process(frame)
    frame.flags.writeable = True
    frame = draw_landmarks(frame, results)
    return frame
    

def draw_landmarks(frame, results):
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        frame,
        results.face_landmarks,
        mp_holistics.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_style
        .get_default_face_mesh_tesselation_style()
    )
    mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_holistics.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_style
        .get_default_pose_landmarks_style()
    )
    mp_drawing.draw_landmarks(
        frame,
        results.left_hand_landmarks,
        mp_holistics.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_style
        .get_default_hand_landmarks_style()
    )
    mp_drawing.draw_landmarks(
        frame,
        results.right_hand_landmarks,
        mp_holistics.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_style
        .get_default_hand_landmarks_style()
    )
    return frame

