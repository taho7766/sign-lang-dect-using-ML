from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_model(sequence_length, feature_length, num_classes):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, feature_length)))
    model.add(LSTM(128, return_sequences=True, activation='relu',))
    model.add(LSTM(64, return_sequences=False, activation='relu', input_shape=(sequence_length, feature_length)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model
    