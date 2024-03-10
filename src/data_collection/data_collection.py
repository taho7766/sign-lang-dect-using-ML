import cv2 as cv
import os
import time
from mediapipe_util import mp_holistic, process_mp, draw_landmarks, extract_keypoints, np

DATA_PATH = '../../MP_DATA'

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


actions = np.array(['hello', 'thanks', 'iloveyou'])
num_sequences = 30
num_frames = 30

for action in actions:
    for sequence in range(num_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass
cap = cv.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        perform_countdown(cap, 3, f'{action} Sequence: 0')
        for sequence in range(num_sequences):
            for frame_num in range(num_frames):
                if frame_num == 0:
                    perform_countdown(cap, 2, f'{action}: {sequence}')
                valid_frame, frame = cap.read()
                image, results = process_mp(frame, holistic)
                keypoints = extract_keypoints(results)
                if(keypoints.shape != (1704,)):
                    print("INCORRECT KEYPOINT SHAPE")
                    quit()
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)
                draw_landmarks(image, results)
                image = cv.flip(image, 1)
                cv.putText(image, 'action: {} sequence: {} frame_num: {}'.format(action, sequence, frame_num), (15, 12),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 225), 1, cv.LINE_AA)
                cv.imshow('TESTING', image)
                if cv.waitKey(1) == ord('q'):
                    quit()
cap.release()
cv.destroyAllWindows()