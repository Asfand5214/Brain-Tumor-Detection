#Preprocessing
```python
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = '/content/drive/MyDrive/Training'
test_dir = '/content/drive/MyDrive/Testing'

# Image size and batch size
img_size = (150, 150)
batch_size = 32

# Data augmentation and normalization
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2,
                                   shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', subset='training')
validation_generator = train_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', subset='validation')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=img_size, batch_size=1, class_mode='categorical', shuffle=False)
```

#Model Training
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # 4 classes: glioma, meningioma, pituitary, no tumor
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=10, validation_data=validation_generator)
```
# Evaluate the model
```python
loss, accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Visualization and tumor detection
import matplotlib.patches as patches

class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
colors = {'glioma': 'red', 'meningioma': 'yellow', 'pituitary': 'orange'}

def plot_image_with_annotation(image, label, pred, box):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    if label == 'no_tumor' and pred == 'no_tumor':
        plt.text(10, 10, '✔️', color='green', fontsize=20)
    else:
        rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=2, edgecolor=colors.get(pred, 'blue'), facecolor='none')
        ax.add_patch(rect)
    plt.title(f"True: {label} - Pred: {pred}")
    plt.show()

for i in range(len(test_generator)):
    img, label = test_generator.next()
    prediction = model.predict(img)
    pred_class = class_names[np.argmax(prediction)]
    true_class = class_names[np.argmax(label)]
    img = np.squeeze(img)
    plot_image_with_annotation(img, true_class, pred_class, [10, 10, 130, 130])  # Example bounding box, replace with actual if available

```
#Distribution Graph
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Get class labels and their corresponding indices
class_labels = train_generator.class_indices
class_labels = {v: k for k, v in class_labels.items()}

# Get the class indices for each sample in the training data
train_labels = train_generator.labels

# Map class indices to class names
train_class_names = [class_labels[idx] for idx in train_labels]

# Calculate the distribution of each class in the training dataset
train_distribution = pd.Series(train_class_names).value_counts(normalize=True) * 100

# Plot the distribution
plt.figure(figsize=(8, 6))
train_distribution.plot(kind='bar', color=['red', 'yellow', 'green', 'orange'])
plt.xlabel('Cancer Type')
plt.ylabel('Percentage')
plt.title('Distribution of Cancer Types in Training Dataset')
plt.show()

```

#Testing
#Image PreProcessing
```python
import cv2
import numpy as np

def preprocess_image(image_path, target_size=(150, 150)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image
```

#Predict and Encircle the Tumor (insert the test image here)
```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
colors = {'glioma': 'red', 'meningioma': 'yellow', 'pituitary': 'orange'}

def plot_image_with_annotation(image, label, pred):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    if label == 'no_tumor' and pred == 'no_tumor':
        plt.text(10, 10, '✔️', color='green', fontsize=20)
    else:
        rect = patches.Rectangle((30, 30), 90, 90, linewidth=2, edgecolor=colors.get(pred, 'blue'), facecolor='none')
        ax.add_patch(rect)

    plt.title(f"True: {label} - Pred: {pred}")
    plt.show()

def predict_and_visualize(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    pred_class = class_names[np.argmax(prediction)]

    # Load the original image for visualization
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Assume the true class is 'unknown' for now as we are just testing
    true_class = 'unknown'

    plot_image_with_annotation(original_image, true_class, pred_class)

# Test the function with an example image
test_image_path = '/content/drive/MyDrive/Testing/pituitary/Te-pi_0014.jpg'  # Replace with actual test image path
predict_and_visualize(test_image_path)

```
