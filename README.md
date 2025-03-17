# Fashion Classification Project

Description of the project.

"""
Fashion Product Classification Project Improvement Plan

1. Code Organization and Documentation
   - Add comprehensive docstrings and comments
   - Restructure code into modular components
   - Implement proper logging

2. Enhanced EDA
   - Analyze class distributions and imbalances
   - Explore relationships between features
   - Visualize sample images from different categories
   - Analyze missing values and data quality

3. Improved Model Architecture
   - Implement more advanced architectures (EfficientNet)
   - Add proper regularization techniques
   - Implement learning rate scheduling
   - Add early stopping and model checkpointing

4. Comprehensive Evaluation
   - Implement confusion matrices for each task
   - Calculate precision, recall, and F1 scores
   - Visualize model predictions
   - Implement cross-validation

5. Robust API
   - Add proper error handling
   - Implement request validation
   - Add API documentation with Swagger
   - Implement rate limiting and security features

6. Enhanced UI
   - Create a more responsive and user-friendly interface
   - Add visualization of model confidence
   - Implement drag-and-drop functionality
   - Add sample images for testing
"""


# Fashion Product Classification

# output

<img src="front1.png" alt="workflow" width="70%">

# Demo:

<img src="front2.png" alt="workflow" width="70%">

<img src="output.png" alt="workflow" width="70%">

<img src="samples.png" alt="workflow" width="70%">


## Project Setup

### Prerequisites
- Ensure you have Anaconda installed.
- Install necessary dependencies in a Conda environment.

### Steps to Set Up and Run the Project

#### 1. Navigate to the Project Directory
```sh
cd "C:\Users\Kushal S\codemonk assessment fashion"
```

#### 2. Activate the Conda Environment
```sh
conda activate codemonk
```

#### 3. Open the Project in VS Code
```sh
code .
```

#### 4. Run the Template Script to Set Up the Project
```sh
python template.py
```
**Expected Output:**
```
✅ Directories created successfully.
✅ Created: fashion-classification/templates/index.html
✅ Created: fashion-classification/improved_fashion_classification.py
✅ Created: fashion-classification/improved_app.py
✅ Created: fashion-classification/requirements.txt
✅ Created: fashion-classification/README.md
🎉 Project setup completed!
```
# Fashion Product Classification Project

## Step-by-Step Guide to Run the Fashion Product Classification Project

Follow these detailed instructions to set up and run the improved fashion product classification project from start to finish.

## 1. Environment Setup

First, let's set up a Python environment for the project:

```sh
conda activate codemonk
```

## 2. Install Dependencies

Create a `requirements.txt` file and install the dependencies:

```sh
pip install -r requirements.txt
```

## 3. Project Structure

Create the following directory structure:

```plaintext
fashion-classification/
├── data/                      # Will contain the dataset
├── models/                    # Will store trained models
├── notebooks/                 # For Jupyter notebooks
├── static/                    # Static files for the web app
├── templates/                 # HTML templates
│   └── index.html             # Improved index.html
├── uploads/                   # For uploaded images
├── sample_images/             # Sample images for testing
├── improved_fashion_classification.py  # Main training script
├── improved_app.py            # Flask API
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation
```

Create the necessary directories:

```sh
mkdir -p data/images models notebooks static templates uploads sample_images
```

## 4. Download and Prepare the Dataset

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset):

```sh
pip install kaggle
kaggle datasets download -d paramaggarwal/fashion-product-images-dataset
unzip fashion-product-images-dataset.zip -d data/
```

## download and load dataset in data folder for running model training train.py

```bash
download dataset from kaggle by above link and add in data folder
```

## 5. Prepare Sample Images

```sh
cp data/fashion-dataset/images/10001.jpg sample_images/
cp data/fashion-dataset/images/10002.jpg sample_images/
cp data/fashion-dataset/images/10003.jpg sample_images/
cp data/fashion-dataset/images/10004.jpg sample_images/
```

## 6. run the Code Files one by one for full execution

# run train.py to run code and it will ask user weather to retrain model
```bash
 train.py
 ```
# do you want to retrain the model

```bash
do you want to retrain model (yes/no):
```
# if "YES"
```bash
model training again
```

# save models a
```models
best_fashion_model.pt
```
```models
label_encoders.pkl
```

# no
```bash
model evaluation started
```
# run app.py do see demonstration of this project

 ```bash
app.py
```
# explore webpage and add sample images and predict

# use templates folder to rendertemplate using flask

```note
Save `templates/index.html`
```

## 7. if you want to evalute model after training model

```bash
evaluate.py
```
# give output graphs and all output samples in terminal for code running demonstration

# if u want to save all demonstration files without training and evaluating 
```bash
save_visualization.py
```
# it will save all procedure output in 
```folder
visualization_results
```

## . Run the Flask API

```sh
python app.py
```

##  Access the Web Interface

Open: `http://localhost:5000`

##  Test with External Images

1. Download fashion product images
2. Upload them via the web interface


## 13. Optional: Deploy to Production

```sh
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 improved_app:app
```

To use Docker:

```sh
docker build -t fashion-classification .
docker run -p 5000:5000 fashion-classification
```

## 14. Project Enhancements

1. **Improve Model Performance**:
   - Fine-tune hyperparameters
   - Try different architectures
   - Implement advanced data augmentation

2. **Expand Web Features**:
   - User authentication
   - History of classifications

3. **Enhance API**:
   - Batch processing
   - Authentication

## Summary

This project provides:
- A well-structured ML pipeline
- A deep learning model for classification
- A robust Flask API
- A user-friendly web interface