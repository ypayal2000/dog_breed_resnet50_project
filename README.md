# Dog Breed Classification using ResNet50 Transfer Learning

This project is a practical implementation of **transfer learning** using **ResNet50** for multi-class dog breed classification. The model is trained to predict different dog breeds from an input image.

The project includes dataset loading, preprocessing, ResNet50 model building, model training, accuracy/loss graph plotting, prediction using a saved model, and a Streamlit-based user interface.

---

## Project Objective

The objective of this project is to build a deep learning model that can classify dog images into different dog breed categories.

Example:

```text
Input: Dog image
Output: Predicted dog breed with confidence score
```

---

## Tech Stack

```text
Python
TensorFlow / Keras
ResNet50
Transfer Learning
Matplotlib
NumPy
Pandas
Pillow
Streamlit
```

---

## Project Workflow

```text
Dog Breed Dataset
        ↓
Load Images from Train/Test Folders
        ↓
Resize Images to 224 x 224
        ↓
Apply ResNet50 Preprocessing
        ↓
Load Pre-trained ResNet50 Model
        ↓
Add Custom Classification Layers
        ↓
Train Model
        ↓
Save Model and Class Names
        ↓
Plot Accuracy and Loss Graphs
        ↓
Predict Dog Breed
        ↓
Show Result in Streamlit UI
```

---

## Dataset

This project uses a dog breed dataset such as the **Stanford Dogs Dataset**.

Recommended dataset structure:

```text
data/
│
├── train/
│   ├── Chihuahua/
│   ├── Golden_retriever/
│   ├── Labrador_retriever/
│   └── ...
│
└── test/
    ├── Chihuahua/
    ├── Golden_retriever/
    ├── Labrador_retriever/
    └── ...
```

Each dog breed should have its own folder. The folder name is treated as the class label.

---

## Project Structure

```text
dog_breed_resnet50_project/
│
├── data/
│   ├── train/
│   └── test/
│
├── models/
│   ├── dog_breed_resnet50.h5
│   └── class_names.json
│
├── output/
│   └── plots/
│       ├── accuracy_plot.png
│       └── loss_plot.png
│
├── src/
│   ├── check_dataset.py
│   ├── data_loader.py
│   ├── model_builder.py
│   └── plot_utils.py
│
├── test_images/
│   └── dog.jpg
│
├── app.py
├── predict.py
├── train.py
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd dog_breed_resnet50_project
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

For Mac/Linux:

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Requirements

Your `requirements.txt` should contain:

```text
tensorflow
pandas
numpy
matplotlib
pillow
scikit-learn
streamlit
```

---

## Step 1: Check Dataset

Before training, check whether the dataset is properly arranged.

Run:

```bash
python src/check_dataset.py
```

Expected output:

```text
Dataset Summary
Total train classes: 120
Total test classes: 120
Total train images: 12000
Total test images: 8580
```

This step verifies:

```text
Train folder exists
Test folder exists
Class folders are available
Images are present inside each class folder
Train and test classes are matching
```

---

## Step 2: Load and Preprocess Dataset

The dataset is loaded using TensorFlow:

```python
tf.keras.utils.image_dataset_from_directory
```

Images are resized to:

```text
224 x 224 x 3
```

This is required because ResNet50 expects input images in this format.

ResNet50 preprocessing is applied using:

```python
preprocess_input
```

This ensures that the input images are processed in the same format as the original ResNet50 training.

---

## Step 3: Build ResNet50 Transfer Learning Model

The model uses pre-trained ResNet50 as a base model:

```text
ResNet50 without top layer
        ↓
GlobalAveragePooling2D
        ↓
Dense Layer
        ↓
Dropout
        ↓
Dense Softmax Output Layer
```

ResNet50 is loaded with ImageNet weights:

```python
weights="imagenet"
```

The original top layer is removed:

```python
include_top=False
```

The ResNet50 base model is frozen:

```python
base_model.trainable = False
```

This means only the newly added classification layers are trained.

---

## Step 4: Train the Model

Run:

```bash
python train.py
```

The training script performs the following tasks:

```text
Loads train and test datasets
Builds the ResNet50 transfer learning model
Trains the model
Saves the trained model
Saves class names
Generates accuracy and loss graphs
```

After training, the model is saved at:

```text
models/dog_breed_resnet50.h5
```

Class names are saved at:

```text
models/class_names.json
```

---

## Step 5: Training Graphs

After training, two graphs are generated:

```text
output/plots/accuracy_plot.png
output/plots/loss_plot.png
```

### Accuracy Graph

This graph compares:

```text
Training accuracy
Validation accuracy
```

It helps identify whether the model is learning correctly.

### Loss Graph

This graph compares:

```text
Training loss
Validation loss
```

It helps identify overfitting or underfitting.

---

## Step 6: Predict Dog Breed from Image

Place a test image inside:

```text
test_images/dog.jpg
```

Then run:

```bash
python predict.py
```

Example output:

```text
Predicted Breed: Golden_retriever
Confidence: 87.45 %

Top 5 Predictions:
Golden_retriever: 87.45%
Labrador_retriever: 6.23%
Cocker_spaniel: 2.14%
Kuvasz: 1.31%
Clumber: 0.89%
```

---

## Step 7: Run Streamlit UI

Run:

```bash
streamlit run app.py
```

The Streamlit app allows users to:

```text
Upload dog image
View uploaded image
Predict dog breed
View confidence score
View top 5 predictions
```

---

## Model Output

The model gives probability scores for all dog breed classes.

Example:

```text
Golden_retriever: 92.45%
Labrador_retriever: 4.12%
Cocker_spaniel: 1.58%
```

The breed with the highest probability is selected as the final prediction.

---

## Why Transfer Learning is Used

Training a deep CNN from scratch requires:

```text
Large dataset
High GPU power
Long training time
```

ResNet50 is already trained on ImageNet and has learned useful visual features such as:

```text
Edges
Textures
Shapes
Fur patterns
Eyes
Ears
Object parts
```

Using transfer learning, we reuse these learned features and train only the final classification layers for dog breed prediction.

---

## Key Concepts Used

### Transfer Learning

Using knowledge from a pre-trained model and applying it to a new task.

### ResNet50

A deep convolutional neural network with 50 layers. It uses residual connections to solve the vanishing gradient problem.

### Global Average Pooling

Converts feature maps into a single feature vector and reduces the number of parameters.

### Dropout

Randomly disables neurons during training to reduce overfitting.

### Softmax

Converts model outputs into class probabilities.

### Categorical Crossentropy

Loss function used for multi-class classification when labels are one-hot encoded.

---

## Results

The model produces:

```text
Trained ResNet50 model
Dog breed prediction
Confidence score
Top 5 predicted breeds
Training accuracy graph
Training loss graph
Streamlit UI
```

---

## Future Improvements

Possible improvements for this project:

```text
Fine-tune top layers of ResNet50
Add data augmentation
Add confusion matrix
Add classification report
Deploy Streamlit app
Use EfficientNet or MobileNetV2
Add Grad-CAM visualization
Add model versioning with MLflow
```

---

## How to Explain This Project in Interview

This project uses transfer learning with ResNet50 to classify dog breeds. The dataset is arranged in train and test folders, where each dog breed is stored in a separate folder. Images are resized to 224 x 224 and preprocessed using ResNet50 preprocessing.

The ResNet50 base model is loaded with ImageNet weights and its original classification layer is removed. A custom classification head is added using GlobalAveragePooling2D, Dense, Dropout, and Softmax layers. During the first training phase, the ResNet50 base is frozen and only the custom layers are trained.

After training, the model and class names are saved. Accuracy and loss graphs are generated using Matplotlib. A prediction script and Streamlit UI are created to upload a dog image and predict the breed with confidence score and top 5 predictions.

---

## Author

```text
Developed as a practical deep learning transfer learning project using ResNet50.
```
