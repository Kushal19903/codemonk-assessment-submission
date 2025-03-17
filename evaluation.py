import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import joblib
import logging
from PIL import Image
import gc  # Garbage collector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 8  # Further reduced batch size to avoid memory issues

def load_label_encoders(encoders_path='label_encoders.pkl'):
    """Load the label encoders from file."""
    if os.path.exists(encoders_path):
        label_encoders = joblib.load(encoders_path)
        num_classes = {task: len(encoder.classes_) for task, encoder in label_encoders.items()}
        logger.info(f"Label encoders loaded from {encoders_path}")
        logger.info(f"Number of classes: {num_classes}")
        return label_encoders, num_classes
    else:
        logger.error(f"Label encoders file not found at {encoders_path}")
        return None, None

def load_model(model_path='best_fashion_model.h5'):
    """Load the trained model from file."""
    if os.path.exists(model_path):
        # Load model with compile=False to avoid metrics warning
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Recompile the model
        model.compile(
            optimizer='adam',
            loss={task: 'categorical_crossentropy' for task in ['articleType', 'baseColour', 'season', 'gender']},
            metrics={task: ['accuracy'] for task in ['articleType', 'baseColour', 'season', 'gender']}
        )
        
        logger.info(f"Model loaded from {model_path}")
        return model
    else:
        logger.error(f"Model file not found at {model_path}")
        return None

def load_data(styles_file):
    """Load the dataset."""
    logger.info(f"Loading dataset from {styles_file}")
    
    try:
        # Load the CSV file
        df = pd.read_csv(styles_file, on_bad_lines='skip')
        logger.info(f"Dataset loaded with {len(df)} samples")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error

def preprocess_data(df, images_dir, label_encoders):
    """Preprocess the data for evaluation."""
    logger.info("Preprocessing data")
    
    # Add image path column
    df['image_path'] = df['id'].apply(lambda x: os.path.join(images_dir, f"{x}.jpg"))
    
    # Filter out rows with missing image files - check only a sample to save time
    sample_size = min(1000, len(df))
    sample_df = df.sample(sample_size, random_state=42)
    valid_images = sample_df['image_path'].apply(os.path.exists)
    valid_ratio = valid_images.mean()
    
    logger.info(f"Sampled {sample_size} images, {valid_ratio:.2%} exist")
    
    # Handle missing values
    for col in ['baseColour', 'season', 'usage', 'year']:
        if col in df.columns and df[col].isnull().sum() > 0:
            # For categorical columns, fill with the most common value
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Encode categorical labels
    for column in ['articleType', 'baseColour', 'season', 'gender']:
        if column in label_encoders:
            # Create a mapping dictionary for efficiency
            mapping = {cls: idx for idx, cls in enumerate(label_encoders[column].classes_)}
            # Apply mapping with a default value for unknown classes
            df[column] = df[column].map(lambda x: mapping.get(x, 0))
    
    logger.info("Data preprocessing completed")
    return df

def evaluate_model_in_batches(model, df, label_encoders, num_classes, batch_size=8, max_samples=100):
    """
    Evaluate the model on a subset of data in small batches to avoid memory issues.
    
    Args:
        model: The trained model
        df: DataFrame with image paths and labels
        label_encoders: Dictionary of label encoders
        num_classes: Dictionary with number of classes for each task
        batch_size: Size of batches to process
        max_samples: Maximum number of samples to evaluate
    """
    logger.info(f"Evaluating model on {min(max_samples, len(df))} samples with batch size {batch_size}")
    
    # Sample data to evaluate
    if len(df) > max_samples:
        eval_df = df.sample(max_samples, random_state=42)
    else:
        eval_df = df
    
    # Initialize arrays to store true and predicted labels
    true_labels = {task: [] for task in num_classes.keys()}
    pred_labels = {task: [] for task in num_classes.keys()}
    
    # Process in small batches
    for i in range(0, len(eval_df), batch_size):
        batch_df = eval_df.iloc[i:i+batch_size]
        
        # Load and preprocess images
        batch_images = []
        valid_indices = []  # Keep track of which rows have valid images
        
        for j, (_, row) in enumerate(batch_df.iterrows()):
            try:
                img_path = row['image_path']
                if os.path.exists(img_path):
                    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
                    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                    batch_images.append(img_array)
                    valid_indices.append(j)
                else:
                    # Skip this sample
                    continue
            except Exception as e:
                logger.warning(f"Error processing image {img_path}: {str(e)}")
                continue
        
        if not batch_images:
            continue
            
        # Convert to numpy array
        batch_images = np.array(batch_images)
        
        try:
            # Get predictions
            predictions = model.predict(batch_images, verbose=0)
            
            # Store true and predicted labels
            for idx, j in enumerate(valid_indices):
                row = batch_df.iloc[j]
                for k, task in enumerate(num_classes.keys()):
                    true_labels[task].append(row[task])
                    pred_labels[task].append(np.argmax(predictions[k][idx]))
            
            # Clear memory
            del batch_images
            del predictions
            gc.collect()
            
            # Print progress
            logger.info(f"Processed {min(i + batch_size, len(eval_df))}/{len(eval_df)} samples")
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            continue
    
    # Evaluate each task
    for task in num_classes.keys():
        if not true_labels[task]:
            logger.warning(f"No valid data for task: {task}")
            continue
            
        print(f"\nEvaluation for {task}:")
        
        # Convert lists to numpy arrays
        y_true = np.array(true_labels[task])
        y_pred = np.array(pred_labels[task])
        
        try:
            # Calculate classification report
            class_names = label_encoders[task].classes_
            print(classification_report(y_true, y_pred, target_names=class_names[:10]))  # Show only first 10 classes
            
            # Plot confusion matrix for top classes
            plt.figure(figsize=(12, 10))
            
            # Get top 10 most frequent classes
            class_counts = np.bincount(y_true, minlength=len(class_names))
            top_classes_idx = np.argsort(class_counts)[-10:]
            
            # Filter data for top classes
            mask = np.isin(y_true, top_classes_idx)
            if np.any(mask):  # Check if any data points match the top classes
                y_true_filtered = y_true[mask]
                y_pred_filtered = y_pred[mask]
                
                # Create confusion matrix
                cm = confusion_matrix(y_true_filtered, y_pred_filtered)
                class_names_filtered = class_names[top_classes_idx]
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=class_names_filtered, yticklabels=class_names_filtered)
                plt.title(f'Confusion Matrix for {task} (Top 10 Classes)')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.xticks(rotation=90)
                plt.yticks(rotation=0)
                plt.tight_layout()
                plt.show()
            else:
                logger.warning(f"No data points for top classes in task: {task}")
        except Exception as e:
            logger.error(f"Error evaluating task {task}: {str(e)}")

def plot_sample_predictions(model, df, label_encoders, num_samples=3):
    """
    Plot sample predictions from the model.
    
    Args:
        model: The trained model
        df: DataFrame with image paths
        label_encoders: Dictionary of label encoders
        num_samples: Number of samples to display
    """
    # Get random samples
    samples = df.sample(min(num_samples, len(df)), random_state=42)
    
    for _, row in samples.iterrows():
        img_path = row['image_path']
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
            
        try:
            # Load and preprocess the image
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make predictions
            predictions = model.predict(img_array, verbose=0)
            
            # Process predictions
            results = {}
            for i, task in enumerate(label_encoders.keys()):
                predicted_class_idx = np.argmax(predictions[i][0])
                predicted_class = label_encoders[task].inverse_transform([predicted_class_idx])[0]
                confidence = float(predictions[i][0][predicted_class_idx])
                results[task] = {
                    'class': predicted_class,
                    'confidence': confidence
                }
                
            # Display the image and predictions
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            plt.axis('off')
            
            # Create prediction text
            pred_text = "\n".join([
                f"{task}: {pred['class']} ({pred['confidence']:.2f})"
                for task, pred in results.items()
            ])
            
            # Get actual values
            actual_values = {}
            for task in label_encoders.keys():
                actual_idx = row[task]
                try:
                    actual_class = label_encoders[task].inverse_transform([actual_idx])[0]
                    actual_values[task] = actual_class
                except:
                    actual_values[task] = "Unknown"
                
            actual_text = "\n".join([
                f"{task}: {value}"
                for task, value in actual_values.items()
            ])
            
            plt.title(f"Predictions:\n{pred_text}\n\nActual:\n{actual_text}", fontsize=12)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {str(e)}")

def save_model_summary(model, output_file='model_summary.txt'):
    """Save model summary to a text file."""
    # Redirect stdout to a string
    import io
    import sys
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    
    # Print model summary
    model.summary()
    
    # Get the string and restore stdout
    summary = new_stdout.getvalue()
    sys.stdout = old_stdout
    
    # Write to file with UTF-8 encoding
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    logger.info(f"Model summary saved to {output_file}")

def main():
    # Define paths - adjust these to match your actual paths
    data_dir = r"C:\Users\Kushal S\Desktop\codemonk2\fashion-classification\data\fashion-dataset"
    images_dir = os.path.join(data_dir, "images")
    styles_file = os.path.join(data_dir, "styles.csv")
    
    # Load label encoders and model
    label_encoders, num_classes = load_label_encoders()
    if not label_encoders:
        logger.error("Failed to load label encoders. Exiting.")
        return
        
    model = load_model()
    if not model:
        logger.error("Failed to load model. Exiting.")
        return
    
    try:
        # Save model summary
        save_model_summary(model)
        
        # Load and preprocess data
        df = load_data(styles_file)
        if df.empty:
            logger.error("Failed to load data. Exiting.")
            return
            
        df = preprocess_data(df, images_dir, label_encoders)
        
        # Evaluate model in batches with a very small number of samples first
        evaluate_model_in_batches(model, df, label_encoders, num_classes, batch_size=BATCH_SIZE, max_samples=100)
        
        # Plot sample predictions
        plot_sample_predictions(model, df, label_encoders, num_samples=3)
        
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

