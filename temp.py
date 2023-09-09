# This is the train script for CNN - Deep_Learning_Basics
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define the root directory where your data is located
data_dir = r"C:\Users\jaiva\Desktop\kode\image\data"  # Replace with your data directory

# Define the batch size and image size
batch_size = 64
img_height = 180
img_width = 180

# Create an ImageDataGenerator for data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split the data into training and validation sets
)

# Load and prepare the training dataset
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse',  # For integer-encoded labels
    subset='training'  # Specify that this is the training dataset
)

# Load and prepare the validation dataset
validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation'  # Specify that this is the validation dataset
)

# Create the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(train_generator.class_indices))  # Output layer with the number of classes
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10  # You can adjust the number of epochs as needed
)

# Save the model
model.save('cnn_model.h5')

# Plot the training and validation accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(validation_generator, verbose=2)
print(f"Accuracy on Validation Dataset: {test_acc}")

plt.subplot(1, 2, 2)
plt.bar(['Validation'], [test_acc], color='skyblue', label='Validation Accuracy')
plt.ylim([0, 1])
plt.title('Validation Accuracy')
plt.legend(loc='lower right')
plt.show()
