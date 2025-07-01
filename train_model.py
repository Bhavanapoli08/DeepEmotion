from data_loader import load_data
from model import create_model
import tensorflow as tf
import os

train_dir = 'data/train/'
test_dir = 'data/test/'

# Load data and class names
train_data, test_data, class_names = load_data(train_dir, test_dir)
num_classes = len(class_names)

# Normalize pixel values
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
test_data = test_data.map(lambda x, y: (normalization_layer(x), y))

# Performance optimization
AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_data = test_data.cache().prefetch(buffer_size=AUTOTUNE)

# Create and compile model
model = create_model(input_shape=(48, 48, 1), num_classes=num_classes)

# Train the model
model.fit(
    train_data,
    epochs=50,
    validation_data=test_data
)

# Save the model
os.makedirs("models", exist_ok=True)
model.save('models/emotion_model_final.h5')
print("✅ Model saved as 'models/emotion_model_final.h5'")
