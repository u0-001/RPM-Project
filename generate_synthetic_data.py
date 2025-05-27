import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np

# Load preprocessed data
data = pd.read_csv('C:/RPM/data/preprocessed_vitals.csv')
vital_columns = ['heartrate', 'sao2', 'respiration', 'systemicsystolic', 'systemicdiastolic']

# Adjust columns
available_columns = data.columns.tolist()
if 'systemicsystolic' not in available_columns:
    if 'noninvasivesystolic' in available_columns:
        vital_columns[3] = 'noninvasivesystolic'
    elif 'systolic' in available_columns:
        vital_columns[3] = 'systolic'
if 'systemicdiastolic' not in available_columns:
    if 'noninvasivediastolic' in available_columns:
        vital_columns[4] = 'noninvasivediastolic'
    elif 'diastolic' in available_columns:
        vital_columns[4] = 'diastolic'

print("Using columns for GAN:", vital_columns)
real_data = data[vital_columns].values

# Define generator
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=100),
        layers.Dense(64, activation='relu'),
        layers.Dense(5, activation='sigmoid')
    ])
    return model

# Define discriminator
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(5,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Compile models
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer='adam', loss='binary_crossentropy')
discriminator.trainable = False

# GAN model
gan_input = layers.Input(shape=(100,))
fake_data = generator(gan_input)
gan_output = discriminator(fake_data)
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Train GAN
epochs = 200
batch_size = 32
for epoch in range(epochs):
    idx = np.random.randint(0, real_data.shape[0], batch_size)
    real_batch = real_data[idx]
    noise = np.random.normal(0, 1, (batch_size, 100))
    fake_batch = generator.predict(noise, verbose=0)
    d_loss_real = discriminator.train_on_batch(real_batch, np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch(fake_batch, np.zeros((batch_size, 1)))
    noise = np.random.normal(0, 1, (batch_size, 100))
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, D Loss Real: {d_loss_real}, D Loss Fake: {d_loss_fake}, G Loss: {g_loss}")

# Generate synthetic data
noise = np.random.normal(0, 1, (250, 100))
synthetic_data = generator.predict(noise)
synthetic_df = pd.DataFrame(synthetic_data, columns=vital_columns)

# Save synthetic data
synthetic_df.to_csv('C:/RPM/data/synthetic_vitals.csv', index=False)

# Combine with preprocessed data
preprocessed_data = pd.read_csv('C:/RPM/data/preprocessed_vitals.csv')
# Concatenate vital signs
combined_vitals = pd.concat([preprocessed_data[vital_columns], synthetic_df], ignore_index=True)
# Add non-vital columns for preprocessed data, NaN for synthetic
combined_data = pd.DataFrame({
    'patientunitstayid': preprocessed_data['patientunitstayid'].tolist() + [None] * 250,
    'observationoffset': preprocessed_data['observationoffset'].tolist() + [None] * 250
})
combined_data[vital_columns] = combined_vitals

# Save combined data
combined_data.to_csv('C:/RPM/data/combined_vitals.csv', index=False)

print("Synthetic data saved to C:/RPM/data/synthetic_vitals.csv")
print("Combined data saved to C:/RPM/data/combined_vitals.csv")
print("Combined rows:", len(combined_data))