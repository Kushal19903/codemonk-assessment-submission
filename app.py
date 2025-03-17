# Flask API script
"""
Fashion Product Classification API

This Flask application serves a deep learning model that classifies fashion product images
into multiple categories: article type, color, season, and gender.

Author: [Your Name]
Date: [Current Date]
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import uuid
import joblib
import logging
from werkzeug.utils import secure_filename
from functools import wraps
import time
from flask_cors import CORS
from PIL import Image
import io
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
#app.config['MODEL_PATH'] = 'fashion_product_model (1).h5'
app.config['MODEL_PATH'] = 'best_fashion_model.h5'    
app.config['ENCODERS_PATH'] = 'label_encoders.pkl'
app.config['SAMPLE_IMAGES_FOLDER'] = 'sample_images'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SAMPLE_IMAGES_FOLDER'], exist_ok=True)

# Load the model and label encoders
try:
    logger.info(f"Loading model from {app.config['MODEL_PATH']}")
    model = load_model(app.config['MODEL_PATH'])
    
    logger.info(f"Loading label encoders from {app.config['ENCODERS_PATH']}")
    label_encoders = joblib.load(app.config['ENCODERS_PATH'])
    
    logger.info("Model and label encoders loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or encoders: {str(e)}")
    raise

# Utility functions
def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(img_path):
    """
    Preprocess an image for model prediction.
    
    Args:
        img_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Preprocessed image array
    """
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def predict_fashion_product(img_array):
    """
    Make predictions on a preprocessed image.
    
    Args:
        img_array (numpy.ndarray): Preprocessed image array
        
    Returns:
        dict: Prediction results for each task
    """
    try:
        # Make predictions
        predictions = model.predict(img_array)
        
        # Process predictions
        results = {}
        tasks = list(label_encoders.keys())
        
        for i, task in enumerate(tasks):
            predicted_class_idx = np.argmax(predictions[i][0])
            predicted_class = label_encoders[task].inverse_transform([predicted_class_idx])[0]
            confidence = float(predictions[i][0][predicted_class_idx])
            
            results[task] = {
                'class': predicted_class,
                'confidence': confidence
            }
            
        return results
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise

# Decorators
def timer(f):
    """Decorator to time API calls."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function {f.__name__} took {end_time - start_time:.2f} seconds to execute")
        return result
    return wrapper

# Routes
@app.route('/')
def home():
    """Render the home page."""
    # Get list of sample images
    sample_images = [f for f in os.listdir(app.config['SAMPLE_IMAGES_FOLDER']) 
                    if allowed_file(f)]
    return render_template('index.html', sample_images=sample_images)

@app.route('/sample_images/<filename>')
def sample_image(filename):
    """Serve sample images."""
    return send_from_directory(app.config['SAMPLE_IMAGES_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
@timer
def predict():
    """
    Predict fashion product attributes from an uploaded image.
    
    Accepts either a file upload or a base64 encoded image.
    """
    try:
        # Check if the post request has the file part
        if 'file' in request.files:
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({"error": "No selected file"}), 400
                
            if file and allowed_file(file.filename):
                # Generate a unique filename
                filename = secure_filename(str(uuid.uuid4()) + '_' + file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                # Save the file
                file.save(file_path)
                logger.info(f"File saved to {file_path}")
                
                # Preprocess the image
                img_array = preprocess_image(file_path)
                
                # Make predictions
                results = predict_fashion_product(img_array)
                
                # Add image path to results
                results['image_path'] = file_path
                
                return jsonify(results)
            else:
                return jsonify({"error": "File type not allowed"}), 400
                
        # Check if base64 image was provided
        elif 'image_data' in request.json:
            try:
                # Decode base64 image
                image_data = request.json['image_data']
                image_data = image_data.split(',')[1] if ',' in image_data else image_data
                
                # Convert to image
                img = Image.open(io.BytesIO(base64.b64decode(image_data)))
                
                # Save the image
                filename = f"{uuid.uuid4()}.jpg"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                img.save(file_path)
                
                # Preprocess the image
                img_array = preprocess_image(file_path)
                
                # Make predictions
                results = predict_fashion_product(img_array)
                
                # Add image path to results
                results['image_path'] = file_path
                
                return jsonify(results)
            except Exception as e:
                logger.error(f"Error processing base64 image: {str(e)}")
                return jsonify({"error": f"Error processing image: {str(e)}"}), 400
        else:
            return jsonify({"error": "No file or image data provided"}), 400
            
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/predict_sample/<filename>', methods=['GET'])
@timer
def predict_sample(filename):
    """Predict on a sample image."""
    try:
        file_path = os.path.join(app.config['SAMPLE_IMAGES_FOLDER'], filename)
        
        if not os.path.exists(file_path):
            return jsonify({"error": "Sample image not found"}), 404
            
        # Preprocess the image
        img_array = preprocess_image(file_path)
        
        # Make predictions
        results = predict_fashion_product(img_array)
        
        # Add image path to results
        results['image_path'] = f"/sample_images/{filename}"
        
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error predicting sample image: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model_loaded": model is not None}), 200

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    return jsonify({"error": "File too large"}), 413

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error"}), 500

# Run the Flask app
if __name__ == '__main__':
    logger.info("Starting Fashion Product Classification API")
    app.run(debug=False, host='0.0.0.0', port=5000)