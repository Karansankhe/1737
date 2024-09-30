import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Simulate some sample data for I/Q signal (replace with real data)
def generate_synthetic_data(num_samples, num_timesteps, num_features, num_classes):
    X = np.random.rand(num_samples, num_timesteps, num_features)  # Random I/Q data
    y = np.random.randint(0, num_classes, num_samples)  # Random modulation classes
    return X, y

# Parameters
num_samples = 10000       # Number of data samples
num_timesteps = 128       # Time steps (sequence length)
num_features = 2          # Features (I/Q channels)
num_classes = 5           # Modulation types (QPSK, 8PSK, etc.)

# Generate synthetic data
X, y = generate_synthetic_data(num_samples, num_timesteps, num_features, num_classes)

# Encode labels (modulation classes)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_one_hot = tf.keras.utils.to_categorical(y_encoded, num_classes)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Build the AMR model
def build_amr_model(input_shape, num_classes):
    model = Sequential()

    # Feature extraction using CNN
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    
    # Temporal analysis using LSTM
    model.add(LSTM(100, return_sequences=False))

    # Classification head
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define model input shape
input_shape = (num_timesteps, num_features)

# Build the model
model = build_amr_model(input_shape, num_classes)

# Print model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Save the model
model.save('amr_dvb_s2x_model.h5')

# Load and use the model for inference
loaded_model = tf.keras.models.load_model('amr_dvb_s2x_model.h5')
predictions = loaded_model.predict(X_test[:5])

# Display predictions
for i, prediction in enumerate(predictions):
    print(f"Sample {i+1}: Predicted Class - {np.argmax(prediction)}, True Class - {np.argmax(y_test[i])}")
