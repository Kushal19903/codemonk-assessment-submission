# Main training script
# Fashion Product Image Classification - Multi-Task Deep Learning Model
# Author: [Your Name]
# Date: [Current Date]

# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import joblib
import logging
from PIL import Image
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# Configuration parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3

class FashionProductClassifier:
    """
    A class for building and training a multi-task fashion product classifier.
    
    This classifier can predict multiple attributes of fashion products:
    - Article type (e.g., T-shirt, shoes)
    - Base color
    - Season
    - Gender
    """
    
    def __init__(self, data_dir, images_dir, styles_file):
        """
        Initialize the classifier with data paths.
        
        Args:
            data_dir (str): Path to the main data directory
            images_dir (str): Path to the directory containing images
            styles_file (str): Path to the CSV file with product metadata
        """
        self.data_dir = data_dir
        self.images_dir = images_dir
        self.styles_file = styles_file
        self.df = None
        self.label_encoders = {}
        self.num_classes = {}
        self.model = None
        self.history = None
        
    def load_data(self):
        """Load and preprocess the dataset."""
        logger.info("Loading dataset from %s", self.styles_file)
        
        # Load the CSV file
        self.df = pd.read_csv(self.styles_file, on_bad_lines='skip')
        logger.info("Dataset loaded with %d samples", len(self.df))
        
        # Add image path column
        self.df['image_path'] = self.df['id'].apply(lambda x: os.path.join(self.images_dir, f"{x}.jpg"))
        
        # Filter out rows with missing image files
        self.df = self.df[self.df['image_path'].apply(os.path.exists)]
        logger.info("%d samples have corresponding image files", len(self.df))
        
        # Handle missing values
        for col in ['baseColour', 'season', 'usage', 'year']:
            if self.df[col].isnull().sum() > 0:
                # For categorical columns, fill with the most common value
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
                
        logger.info("Missing values handled")
        
        return self.df
    
    def perform_eda(self):
        """Perform exploratory data analysis on the dataset."""
        logger.info("Performing exploratory data analysis")
        
        # Display basic information about the dataset
        print("Dataset Information:")
        print(self.df.info())
        
        print("\nSummary Statistics:")
        print(self.df.describe(include='all'))
        
        print("\nMissing Values:")
        print(self.df.isnull().sum())
        
        # Visualize class distributions
        self._plot_class_distribution('articleType', top_n=20)
        self._plot_class_distribution('baseColour', top_n=20)
        self._plot_class_distribution('season')
        self._plot_class_distribution('gender')
        
        # Visualize relationships between variables
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.df[['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season']].apply(lambda x: pd.factorize(x)[0]).corr(), 
                   annot=True, cmap='coolwarm')
        plt.title("Correlation Between Categorical Variables")
        plt.tight_layout()
        plt.show()
        
        # Display sample images
        self._display_sample_images()
        
    def _plot_class_distribution(self, column, top_n=None):
        """
        Plot the distribution of classes for a specific column.
        
        Args:
            column (str): The column name to plot
            top_n (int, optional): Number of top classes to display. If None, display all.
        """
        plt.figure(figsize=(12, 8))
        
        if top_n:
            # Get the top N most frequent classes
            top_classes = self.df[column].value_counts().nlargest(top_n).index
            # Filter the dataframe to include only these classes
            plot_df = self.df[self.df[column].isin(top_classes)]
            sns.countplot(y=column, data=plot_df, order=plot_df[column].value_counts().index)
            plt.title(f"Distribution of Top {top_n} {column} Classes")
        else:
            sns.countplot(y=column, data=self.df, order=self.df[column].value_counts().index)
            plt.title(f"Distribution of {column} Classes")
            
        plt.tight_layout()
        plt.show()
        
        # Print class imbalance statistics
        class_counts = self.df[column].value_counts()
        total = len(self.df)
        print(f"\n{column} Class Distribution:")
        for cls, count in class_counts.items():
            print(f"{cls}: {count} samples ({count/total*100:.2f}%)")
            
    def _display_sample_images(self, n_samples=5):
        """
        Display sample images from each gender category.
        
        Args:
            n_samples (int): Number of samples to display per category
        """
        plt.figure(figsize=(15, 10))
        
        for i, gender in enumerate(self.df['gender'].unique()):
            # Get sample images for this gender
            samples = self.df[self.df['gender'] == gender].sample(min(n_samples, sum(self.df['gender'] == gender)))
            
            for j, (_, row) in enumerate(samples.iterrows()):
                plt.subplot(len(self.df['gender'].unique()), n_samples, i*n_samples + j + 1)
                img = plt.imread(row['image_path'])
                plt.imshow(img)
                plt.title(f"{gender}: {row['articleType']}\n{row['baseColour']}, {row['season']}")
                plt.axis('off')
                
        plt.tight_layout()
        plt.show()
        
    def preprocess_data(self):
        """Preprocess the data for model training."""
        logger.info("Preprocessing data")
        
        # Encode categorical labels
        for column in ['articleType', 'baseColour', 'season', 'gender']:
            le = LabelEncoder()
            self.df[column] = le.fit_transform(self.df[column])
            self.label_encoders[column] = le
            self.num_classes[column] = len(le.classes_)
            
        logger.info("Label encoding completed")
        logger.info("Number of classes: %s", self.num_classes)
        
        # Split the dataset into training and validation sets
        train_df, val_df = train_test_split(self.df, test_size=0.2, random_state=SEED, stratify=self.df['gender'])
        logger.info("Dataset split: %d training samples, %d validation samples", 
                   len(train_df), len(val_df))
        
        return train_df, val_df
    
    def create_data_generators(self, train_df, val_df):
        """
        Create data generators for training and validation.
        
        Args:
            train_df (DataFrame): Training data
            val_df (DataFrame): Validation data
            
        Returns:
            tuple: Training and validation data generators
        """
        logger.info("Creating data generators")
        
        # Create ImageDataGenerator for data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Custom data generator class
        class MultiOutputDataGenerator(tf.keras.utils.Sequence):
            def __init__(self, dataframe, x_col, y_cols, batch_size, img_size, num_classes, 
                         datagen=None, shuffle=True):
                self.dataframe = dataframe
                self.x_col = x_col
                self.y_cols = y_cols
                self.batch_size = batch_size
                self.img_size = img_size
                self.num_classes = num_classes
                self.datagen = datagen
                self.shuffle = shuffle
                self.indexes = np.arange(len(self.dataframe))
                if self.shuffle:
                    np.random.shuffle(self.indexes)
                
            def __len__(self):
                return int(np.ceil(len(self.dataframe) / self.batch_size))
            
            def __getitem__(self, index):
                # Get batch indexes
                batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
                batch_df = self.dataframe.iloc[batch_indexes]
                
                # Initialize batch data
                batch_x = np.zeros((len(batch_df), self.img_size[0], self.img_size[1], 3))
                batch_y = {col: np.zeros((len(batch_df), self.num_classes[col])) for col in self.y_cols}
                
                # Load and preprocess images
                for i, (_, row) in enumerate(batch_df.iterrows()):
                    # Load image
                    img = tf.keras.preprocessing.image.load_img(row[self.x_col], target_size=self.img_size)
                    img = tf.keras.preprocessing.image.img_to_array(img)
                    
                    # Apply data augmentation if available
                    if self.datagen:
                        img = self.datagen.random_transform(img)
                        img = self.datagen.standardize(img)
                    else:
                        img = img / 255.0
                        
                    batch_x[i] = img
                    
                    # One-hot encode labels
                    for col in self.y_cols:
                        batch_y[col][i] = to_categorical(row[col], num_classes=self.num_classes[col])
                
                return batch_x, batch_y
            
            def on_epoch_end(self):
                if self.shuffle:
                    np.random.shuffle(self.indexes)
        
        # Create custom data generators
        train_generator = MultiOutputDataGenerator(
            dataframe=train_df,
            x_col='image_path',
            y_cols=['articleType', 'baseColour', 'season', 'gender'],
            batch_size=BATCH_SIZE,
            img_size=IMG_SIZE,
            num_classes=self.num_classes,
            datagen=train_datagen,
            shuffle=True
        )
        
        val_generator = MultiOutputDataGenerator(
            dataframe=val_df,
            x_col='image_path',
            y_cols=['articleType', 'baseColour', 'season', 'gender'],
            batch_size=BATCH_SIZE,
            img_size=IMG_SIZE,
            num_classes=self.num_classes,
            datagen=None,
            shuffle=False
        )
        
        logger.info("Data generators created")
        
        return train_generator, val_generator
    
    def build_model(self):
        """Build the multi-task deep learning model."""
        logger.info("Building model")
        
        # Load EfficientNetB0 as the base model
        base_model = EfficientNetB0(
            input_shape=(*IMG_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False
            
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        
        # Define output layers for each task
        outputs = []
        output_names = []
        
        for task, num_class in self.num_classes.items():
            output = Dense(num_class, activation='softmax', name=task)(x)
            outputs.append(output)
            output_names.append(task)
            
        # Create the model
        self.model = Model(inputs=base_model.input, outputs=outputs)
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss={task: 'categorical_crossentropy' for task in output_names},
            metrics={task: ['accuracy'] for task in output_names}
        )
        
        # Print model summary
        self.model.summary()
        logger.info("Model built successfully")
        
        return self.model
    
    def train_model(self, train_generator, val_generator, epochs=EPOCHS):
        """
        Train the model.
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs (int): Number of epochs to train
            
        Returns:
            History object containing training metrics
        """
        logger.info("Training model for %d epochs", epochs)
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=REDUCE_LR_PATIENCE,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                'best_fashion_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Model training completed")
        
        return self.history
    
    

# Main execution
if __name__ == "__main__":
    # Define paths
    data_dir = r"C:\Users\Kushal S\Desktop\codemonk2\fashion-classification\data\fashion-dataset"
    images_dir = os.path.join(data_dir, "images")
    styles_file = os.path.join(data_dir, "styles.csv")
    
    # Initialize the classifier
    classifier = FashionProductClassifier(data_dir, images_dir, styles_file)
    
    # Load and explore data
    df = classifier.load_data()
    classifier.perform_eda()
    
    # Preprocess data
    train_df, val_df = classifier.preprocess_data()
    
    # Create data generators
    train_generator, val_generator = classifier.create_data_generators(train_df, val_df)
    
    # Build and train the model
    model = classifier.build_model()
    
    retrain = input("Do you want to retrain the model? (yes/no): ").lower().strip()
    
    if retrain in ['yes', 'y']:
        logger.info("Starting model training process")
    
        history = classifier.train_model(train_generator, val_generator)
        logger.info("Model Training completed")
    
    else:
        logger.info("Pre-trained model Loaded Successfully")