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



def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening camera")
        exit()
    with(mp_holistics.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)) as holistic:
        while(cap.isOpened()):
            valid_data, frame = cap.read()
            if not valid_data:
                print("Ignoring invalid frames... Please wait while camera loads.")
                continue
            frame.flags.writeable = False
            frame = process_mp(frame, holistic)
            cv.imshow("Testing", cv.flip(frame, 1))
            if(cv.waitKey(1) == ord('q')):
                break
    cap.release()
    cv.destroyAllWindows
    
if __name__ == '__main__':
    cProfile.run('main()')