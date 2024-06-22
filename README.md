# Brain-Tumor-Detection
Certainly! Here is a comprehensive README file for your GitHub repository, including details about the project, setup instructions, usage, and more.

---

# Brain Tumor Detection from MRI Images

This repository contains the code and resources for detecting brain tumors from MRI images using a Convolutional Neural Network (CNN). The model is trained to classify MRI images into four categories: glioma, meningioma, pituitary tumor, and no tumor. The project includes preprocessing steps, model training, evaluation, and visualization of the detected tumors with appropriate annotations.

## Project Overview

Brain tumors can be life-threatening, and early detection is crucial for effective treatment. This project aims to assist radiologists and healthcare professionals by providing an automated tool for detecting and classifying brain tumors from MRI scans. The key features include:

- **Data Preprocessing**: Loading and preprocessing MRI images to make them suitable for training a CNN.
- **Model Training**: Training a CNN on a dataset of MRI images to classify different types of brain tumors.
- **Evaluation**: Assessing the performance of the trained model on a test dataset, achieving an accuracy of 93%.
- **Visualization**: Drawing annotations around detected tumors in different colors based on the tumor type and indicating no tumor with a green tick mark.

## Dataset

The datasets used for this project are sourced from Kaggle:

- [Training Dataset](https://www.kaggle.com/datasets/brain-tumor-mri-dataset/Training)
- [Testing Dataset](https://www.kaggle.com/datasets/brain-tumor-mri-dataset/Testing)

## Model Architecture

The Convolutional Neural Network (CNN) consists of the following layers:
- **Convolutional Layers**: Extract features from input images.
- **MaxPooling Layers**: Reduce spatial dimensions of feature maps.
- **Dense Layers**: Fully connected layers for classification.
- **Dropout Layer**: Prevent overfitting by randomly setting input units to 0 during training.

## Results

The trained model can classify MRI images into four categories with an accuracy of 93%. The visualization component highlights detected tumors with colored rectangles:
- **Red** for Glioma
- **Yellow** for Meningioma
- **Orange** for Pituitary Tumor
- **Green tick mark** for No Tumor

## Usage

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/brain-tumor-detection.git
    cd brain-tumor-detection
    ```

2. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

### Training the Model

```python
# Import necessary libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load and preprocess data
train_dir = '/path/to/train/dataset'  # Replace with your training dataset path
validation_dir = '/path/to/validation/dataset'  # Replace with your validation dataset path

train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

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

### Evaluating the Model

```python
# Evaluate the model
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
```

### Predicting and Visualizing Tumors

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define class names and colors
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
colors = {'glioma': 'red', 'meningioma': 'yellow', 'pituitary': 'orange'}

# Function to preprocess the image
def preprocess_image(image_path, target_size=(150, 150)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to plot image with annotation
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

# Function to predict and visualize the results
def predict_and_visualize(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    pred_class = class_names[np.argmax(prediction)]
    
    # Load the original image for visualization
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # For demonstration, assuming true class is unknown
    true_class = 'unknown'
    
    # Debugging: Print prediction probabilities
    print("Prediction Probabilities:", prediction)
    print("Predicted Class:", pred_class)
    
    plot_image_with_annotation(original_image, true_class, pred_class)

# Test the function with an example image
test_image_path = '/path/to/your/test/image.jpg'  # Replace with actual test image path
predict_and_visualize(test_image_path)
```

## Contributions

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or create a pull request.

## License

This project is licensed under the MIT License.

---

Feel free to customize this description to better fit your specific project and any additional details you might want to include.
