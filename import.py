import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Paths
image_dirs = {
    'outright2.csv': r'D:\222\outright2',
    'log_0.csv': r'D:\222\IMG0',
    'evening1.csv': r'C:\Users\User\Downloads\evening1',
    'evening2.csv': r'C:\Users\User\Downloads\evening2',
    'evening3.csv': r'C:\Users\User\Downloads\evening3',
    'night1.csv' : r'C:\Users\User\Downloads\night1',
    'night2.csv' : r'C:\Users\User\Downloads\night2',
    'night3.csv' : r'C:\Users\User\Downloads\night3'
}
csv_files = [r'D:\222\outright2.csv', r'D:\222\log_0.csv', r'C:\Users\User\Downloads\evening1.csv',
 r'C:\Users\User\Downloads\evening2.csv', r'C:\Users\User\Downloads\evening3.csv',r'C:\Users\User\Downloads\night2.csv',r'C:\Users\User\Downloads\night3.csv',r'C:\Users\User\Downloads\night1.csv']
saved_model_dir = r'D:\222\path\to\save\final_model_saved'
tflite_model_path = r'D:\222\path\to\save\mobilenetv2_lane_detection.tflite'

# Load and combine data from CSVs
dataframes = []
for file in csv_files:
    image_dir = image_dirs[os.path.basename(file)]
    df = pd.read_csv(file)
    df['image_path'] = df['Image'].apply(lambda x: os.path.join(image_dir, x))
    df = df[df['image_path'].apply(os.path.exists)]  # Filter images that exist
    dataframes.append(df[['image_path', 'Steering', 'Left_Speed', 'Right_Speed']])
data = pd.concat(dataframes, ignore_index=True)

# Split data
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Preprocessing with Data Augmentation
def load_and_preprocess_image(image_path, steering, left_speed, right_speed):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    
    # Data Augmentation
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

    labels = tf.convert_to_tensor([steering, left_speed, right_speed], dtype=tf.float32)
    return image, labels

def create_dataset(dataframe, batch_size=16):
    dataset = tf.data.Dataset.from_tensor_slices(( 
        dataframe['image_path'].values,
        dataframe['Steering'].values,
        dataframe['Left_Speed'].values,
        dataframe['Right_Speed'].values
    ))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache().shuffle(buffer_size=10000).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# Create datasets
batch_size = 8
train_dataset = create_dataset(train_data, batch_size)
val_dataset = create_dataset(val_data, batch_size)

# Load existing model or create a new one
if os.path.exists(saved_model_dir):
    print("Loading existing model for fine-tuning...")
    model = tf.keras.models.load_model(saved_model_dir)
else:
    print("Creating a new model...")
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze the base model layers
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(3)  # Output for Steering, Left_Speed, Right_Speed
    ])

# Fine-tuning: Unfreeze the top layers of the base model
print("Setting up fine-tuning...")
base_model = model.layers[0]  # สมมติว่า base model เป็นเลเยอร์แรก
base_model.trainable = True
for layer in base_model.layers[:-20]:  # ล็อกเลเยอร์ทั้งหมด ยกเว้น 20 ชั้นสุดท้าย
    layer.trainable = False

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='mse',
    metrics=['mae']
)

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7
)

# Train the model with new data and store history
print("Training the model...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100,
    callbacks=[early_stopping, reduce_lr]
)

# Save the updated model
print("Saving the updated model...")
model.save(saved_model_dir)

# Display the final MSE and MAE from the training history
train_mse = history.history['loss'][-1]  # MSE from training
val_mse = history.history['val_loss'][-1]  # MSE from validation
train_mae = history.history['mae'][-1]  # MAE from training
val_mae = history.history['val_mae'][-1]  # MAE from validation

print(f"Final Training MSE: {train_mse:.4f}")
print(f"Final Validation MSE: {val_mse:.4f}")
print(f"Final Training MAE: {train_mae:.4f}")
print(f"Final Validation MAE: {val_mae:.4f}")

# Convert to TFLite with optimization (quantization)
print("Converting the updated model to TFLite...")
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optimize for size
converter.target_spec.supported_types = [tf.float16]  # Quantization to float16 for smaller size
tflite_model = converter.convert()

# Save the TFLite model to file
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print("TFLite model saved to:", tflite_model_path)
