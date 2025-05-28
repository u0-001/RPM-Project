import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Load data
X_train = np.load('C:/RPM/data/X_train.npy')
y_train = np.load('C:/RPM/data/y_train.npy')
X_val = np.load('C:/RPM/data/X_val.npy')
y_val = np.load('C:/RPM/data/y_val.npy')
X_test = np.load('C:/RPM/data/X_test.npy')

# Define LSTM + Attention model
def build_lstm_attention(input_shape=(10, 5)):
    inputs = layers.Input(shape=input_shape)
    lstm = layers.LSTM(64, return_sequences=True)(inputs)
    attention = layers.Attention()([lstm, lstm])
    flat = layers.Flatten()(attention)
    dense = layers.Dense(32, activation='relu')(flat)
    outputs = layers.Dense(3, activation='softmax')(dense)  # 3 classes
    model = models.Model(inputs, outputs)
    return model

# Build and train
model = build_lstm_attention()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_val, y_val),
    verbose=1
)

# Extract features (remove final layer)
feature_model = models.Model(inputs=model.input, outputs=model.layers[-2].output)
X_train_features = feature_model.predict(X_train, batch_size=16)
X_val_features = feature_model.predict(X_val, batch_size=16)
X_test_features = feature_model.predict(X_test, batch_size=16)

# Save
np.save('C:/RPM/data/X_train_features.npy', X_train_features)
np.save('C:/RPM/data/X_val_features.npy', X_val_features)
np.save('C:/RPM/data/X_test_features.npy', X_test_features)
model.save('C:/RPM/models/lstm_attention.keras')  # Keras format

print("LSTM features saved. Shapes:", X_train_features.shape, X_val_features.shape, X_test_features.shape)