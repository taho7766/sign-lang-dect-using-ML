import numpy as np
import os
import shutil
from model_archetecture import create_model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

class CustomCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('categorical_accuracy') >= 0.95:
            print('\nReached 95% accuracy so cancelling training!')
            self.model.stop_training = True


PROCESSED_PATH = '../../MP_DATA/PROCESSED_DATA'

X = np.load(os.path.join(PROCESSED_PATH, 'X.npy'))
Y = np.load(os.path.join(PROCESSED_PATH, 'Y.npy'))
actions = np.array(['hello', 'thanks', 'iloveyou'])
log_path = os.path.join('log')
tb_callback = TensorBoard(log_dir=log_path)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05)

# from sklearn.metrics import accuracy_score, multilabel_confusion_matrix

# model = load_model('/Users/taewoohong/dev/git/sign-lang-dect-using-ML/src/models/interrupted_model.h5')

# yhat = model.predict(X_train)

# ytrue = np.argmax(Y_train, axis=1).tolist()q
# yhat = np.argmax(yhat, axis=1).tolist()

# print(multilabel_confusion_matrix(ytrue, yhat))
# print(accuracy_score(ytrue, yhat))
custom_callback = CustomCallback()

model = create_model(30, 1650, actions.shape[0])
model.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
try:
    model.fit(X_train,
              Y_train,
              epochs=2000,
              callbacks=[tb_callback, custom_callback])
except KeyboardInterrupt:
    model.save('interrupted_model.h5')
    print('Training interrupted and model saved.')
model.save('interrupted_model.h5')