import cv2 as cv
import os
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from data_collection.mediapipe_util import process_mp, draw_landmarks, extract_keypoints
from scipy import stats

colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv.putText(output_frame, actions[num], (0, 85+num*40), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv.LINE_AA)
        
    return output_frame

actions = np.array(['hello','thanks','iloveyou'])

model_path = '/Users/taewoohong/dev/git/sign-lang-dect-using-ML/src/models/interrupted_model.h5'

LSTM_model = load_model(model_path)
sequence = []
predictions = []
sentence = []
threshold = 0.4

mp_holistic = mp.solutions.holistic

cap = cv.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        valid_frame, frame = cap.read()
        frame = cv.flip(frame, 1)
        
        image, results = process_mp(frame, holistic)
        draw_landmarks(image, results)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        res = None
        
        if len(sequence) == 30:
            res = LSTM_model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))
            
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5: 
                sentence = sentence[-5:]

            # Viz probabilities
            image = prob_viz(res, actions, image, colors)
            
        # if(np.argmax(res) > threshold):
        #     if(len(sentence) > 0):
        #         if(actions[np.argmax(res)] != sentence[-1]):
        #             sentence.append(actions[np.argmax(res)])
        #     else:
        #         sentence.append(actions[np.argmax(res)])
        # if(len(sentence) > 5):
        #     sentence = sentence[-5:]
        
        
        cv.rectangle(image, (0, 0), (640, 40), (245, 177, 16), -1)
        cv.putText(image, ' '.join(sentence), (3, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
        cv.imshow('testing', image)
        
        if(cv.waitKey(1) == ord('q')):
            break
        
cap.release()
cv.destroyAllWindows()
        