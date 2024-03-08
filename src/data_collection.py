import numpy as np
import cv2 as cv
import mediapipe as mp
import cProfile
import os
import time

DATA_PATH = '../MP_DATA'
mp_drawing = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

def process_mp(image, model):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    return image, results
    
def perform_countdown(cap, countdown_duration, output_string):
    countdown_start = time.time()
    while(True):
        retVal, frame = cap.read()
        frame = cv.flip(frame, 1)
        elapsed_time = time.time() - countdown_start
        remaining_time = countdown_duration - elapsed_time
        if(remaining_time > 0):
            cv.putText(frame, f'Starting collection in {remaining_time:.3f} for ' + output_string,
                        (15, 200),
                        cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv.LINE_AA)
            cv.imshow('TESTING', frame)
            if(cv.waitKey(1) == ord('q')):
                quit()
        else:
            break

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
        right_hand_landmark_points = np.zeros(21 * 3)

    if results.left_hand_landmarks:
        left_hand_landmark_points = np.array([[res.x, res.y, res.z, res.visibility] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        left_hand_landmark_points = np.zeros(21 * 3)
        
    if results.pose_landmarks:
        pose_landmark_points = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose_landmark_points = np.zeros(33 * 4)

    if results.face_landmarks:
        face_landmark_points = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()
    else:
        face_landmark_points = np.zeros(468 * 3)
    return np.concatenate([right_hand_landmark_points, left_hand_landmark_points, pose_landmark_points, face_landmark_points])


actions = np.array(['hello', 'thankyou', 'iloveyou'])
num_sequences = 30
num_frames = 30

for action in actions:
    for sequence in range(num_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass
cap = cv.VideoCapture(0)
# exit_loop = perform_countdown(cap=cap, countdown_duration=3, output_string="TESTING 1")
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        perform_countdown(cap, 3, f'{action} Sequence: 0')
        for sequence in range(num_sequences):
            for frame_num in range(num_frames):
                if frame_num == 0:
                    perform_countdown(cap, 2, f'{action}: {sequence}')
                valid_frame, frame = cap.read()
                frame = cv.flip(frame, 1)
                image, results = process_mp(frame, holistic)
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)
                draw_landmarks(image, results)
                cv.putText(image, 'action: {} sequence: {} frame_num: {}'.format(action, sequence, frame_num), (15, 12),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 225), 1, cv.LINE_AA)
                cv.imshow('TESTING', image)
                if cv.waitKey(1) == ord('q'):
                    quit()
cap.release()
cv.destroyAllWindows()