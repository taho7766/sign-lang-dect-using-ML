import numpy as np
import os
from tensorflow.keras.utils import to_categorical

DATA_PATH = "../../MP_DATA"
PROCESSED_PATH = os.path.join(DATA_PATH, "PROCESSED_DATA")
actions = np.array(['hello', 'iloveyou', 'thanks'])
no_sequences = 30
sequence_length = 30

label_map = {label: num for num, label, in enumerate(actions)}

sequences, labels = [],[]
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            file_path = os.path.join(DATA_PATH, action, str(sequence), f'{frame_num}.npy')
            results = np.load(file_path)
            window.append(results)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
Y = to_categorical(labels).astype(int)

os.makedirs(PROCESSED_PATH, exist_ok=True)

np.save(os.path.join(PROCESSED_PATH, 'X.npy'), X)
np.save(os.path.join(PROCESSED_PATH, 'Y.npy'), Y)