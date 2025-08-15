
import numpy as np
import gradio as gr
from tensorflow.keras import models, layers, callbacks
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load CIFAR-10 dataset
(train_image, train_labels), (test_image, test_labels) = cifar10.load_data()
train_image, test_image = train_image / 255.0, test_image / 255.0
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Define class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(train_image)

# Build improved CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define callbacks
cb = [
    callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)
]

# Train the model
model.fit(datagen.flow(train_image, train_labels, batch_size=64),
          epochs=30,
          validation_data=(test_image, test_labels),
          callbacks=cb)

# Load the best saved model
model = load_model("best_model.keras")

# Define Gradio prediction function
def predict_cifar10(img):
    img = img.resize((32, 32))
    img = np.array(img) / 255.0
    img = img.reshape(1, 32, 32, 3)
    prediction = model.predict(img)
    return {class_names[i]: float(prediction[0][i]) for i in range(10)}

# Launch Gradio interface
gr.Interface(
    fn=predict_cifar10,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title=" Image Classifier "
).launch()
