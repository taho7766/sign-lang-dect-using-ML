import numpy as np
import os
from model_archetecture import create_model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

PROCESSED_PATH = '../../MP_DATA/PROCESSED_DATA'
accuracy_threshhold = 0.95
min_epoch = 200

early_stop = EarlyStopping(monitor='categorical_accuracy', mode='max',
                           patience=10,
                           restore_best_weights=True,
                           min_delta=0.001,
                           baseline=accuracy_threshhold)

model_checkpoint = ModelCheckpoint(filepath='best_model.h5', monitor='categorical_accuracy',
                                   mode='max', save_best_only=True, save_weights_only=True)

X = np.load(os.path.join(PROCESSED_PATH, 'X.npy'))
Y = np.load(os.path.join(PROCESSED_PATH, 'Y.npy'))
actions = np.array(['hello', 'thanks', 'iloveyou'])
log_path = os.path.join('log')
tb_callback = TensorBoard(log_dir=log_path)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05)

model = create_model(30, 1704, actions.shape[0])
model.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
try:
    model.fit(X_train, Y_train, epochs=2000, callbacks=[tb_callback])
except KeyboardInterrupt:
    model.save('interrupted_model.h5')
    print('Training interrupted and model saved.')