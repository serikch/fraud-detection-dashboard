from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_file
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import pickle
from werkzeug.utils import secure_filename
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
app.secret_key = 'fraud-detection-hackathon-esilv-2025'

# Configuration
UPLOAD_FOLDER = 'data/uploads'
PROCESSED_FOLDER = 'data/processed'
MODELS_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'csv', 'json', 'pkl'}

# Ensure folders exist
for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, MODELS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize session data
@app.before_request
def initialize_session():
    if 'uploaded_files' not in session:
        session['uploaded_files'] = []
    if 'trained_models' not in session:
        session['trained_models'] = []
    if 'preprocessing_status' not in session:
        session['preprocessing_status'] = {}

@app.route('/')
def index():
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    # Calculate statistics
    stats = {
        'total_transactions': 0,
        'fraud_rate': 0,
        'models_trained': len(session.get('trained_models', [])),
        'files_uploaded': len(session.get('uploaded_files', [])),
        'last_prediction': session.get('last_prediction', 'N/A'),
        'best_model_accuracy': session.get('best_accuracy', 0)
    }
    
    # Try to load Watson X final dataset stats if available
    watson_file = 'data/watsonx_final_dataset.csv'
    if os.path.exists(watson_file):
        try:
            df = pd.read_csv(watson_file, nrows=1000)  # Load sample for quick stats
            stats['total_transactions'] = len(df)
            if 'fraud_label' in df.columns:
                stats['fraud_rate'] = (df['fraud_label'].sum() / len(df)) * 100
        except:
            pass
    
    return render_template('dashboard.html', stats=stats)

@app.route('/upload')
def upload_page():
    uploaded_files = session.get('uploaded_files', [])
    return render_template('upload.html', uploaded_files=uploaded_files)

@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Add to session
        uploaded_files = session.get('uploaded_files', [])
        uploaded_files.append({
            'filename': filename,
            'original_name': file.filename,
            'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'size': os.path.getsize(filepath)
        })
        session['uploaded_files'] = uploaded_files
        
        # Get file info
        file_info = {'filename': filename, 'rows': 0, 'columns': 0}
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath, nrows=5)
                file_info['rows'] = sum(1 for line in open(filepath)) - 1
                file_info['columns'] = len(df.columns)
                file_info['column_names'] = df.columns.tolist()
        except:
            pass
        
        return jsonify({'success': True, 'file_info': file_info})
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/preprocessing')
def preprocessing_page():
    uploaded_files = session.get('uploaded_files', [])
    watson_available = os.path.exists('data/watsonx_final_dataset.csv')
    return render_template('preprocessing.html', 
                         uploaded_files=uploaded_files,
                         watson_available=watson_available)

@app.route('/preprocess', methods=['POST'])
def preprocess_data():
    data = request.json
    file_to_process = data.get('filename')
    use_watson = data.get('use_watson', False)
    
    result = {
        'success': False,
        'message': '',
        'processed_file': None
    }
    
    try:
        if use_watson:
            # Use the Watson X preprocessed file
            source_file = 'data/watsonx_final_dataset.csv'
            if os.path.exists(source_file):
                result['success'] = True
                result['message'] = 'Using Watson X preprocessed dataset'
                result['processed_file'] = 'watsonx_final_dataset.csv'
                session['current_dataset'] = source_file
            else:
                result['message'] = 'Watson X dataset not found'
        else:
            # Simulate preprocessing (in real scenario, implement actual preprocessing)
            result['success'] = True
            result['message'] = f'Preprocessing {file_to_process}... (Feature coming soon)'
            result['processed_file'] = file_to_process
        
        session['preprocessing_status'] = result
        
    except Exception as e:
        result['message'] = str(e)
    
    return jsonify(result)

@app.route('/training')
def training_page():
    processed_files = session.get('preprocessing_status', {})
    return render_template('training.html', processed_files=processed_files)

@app.route('/train_model', methods=['POST'])
def train_model():
    data = request.json
    model_type = data.get('model_type', 'autoai')
    
    result = {
        'success': False,
        'message': '',
        'model_name': None,
        'accuracy': 0
    }
    
    try:
        # Simulate model training (integrate with Watson X AutoAI in production)
        if model_type == 'autoai':
            result['success'] = True
            result['message'] = 'Model training with Watson X AutoAI... (Integration coming soon)'
            result['model_name'] = f'fraud_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            result['accuracy'] = np.random.uniform(0.85, 0.95)  # Simulated accuracy
            
            # Add to trained models
            trained_models = session.get('trained_models', [])
            trained_models.append({
                'name': result['model_name'],
                'type': model_type,
                'accuracy': result['accuracy'],
                'training_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            session['trained_models'] = trained_models
            session['best_accuracy'] = max(m['accuracy'] for m in trained_models)
        else:
            result['message'] = 'Model type not supported yet'
    
    except Exception as e:
        result['message'] = str(e)
    
    return jsonify(result)

@app.route('/models')
def models_page():
    trained_models = session.get('trained_models', [])
    return render_template('models.html', trained_models=trained_models)

@app.route('/upload_model', methods=['POST'])
def upload_model():
    if 'model_file' not in request.files:
        return jsonify({'error': 'No model file provided'}), 400
    
    file = request.files['model_file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename.endswith('.pkl'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(MODELS_FOLDER, filename)
        file.save(filepath)
        
        # Add to trained models
        trained_models = session.get('trained_models', [])
        trained_models.append({
            'name': filename,
            'type': 'uploaded',
            'accuracy': 'Unknown',
            'training_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        session['trained_models'] = trained_models
        
        return jsonify({'success': True, 'model_name': filename})
    
    return jsonify({'error': 'Invalid model file'}), 400

@app.route('/prediction')
def prediction_page():
    trained_models = session.get('trained_models', [])
    return render_template('prediction.html', trained_models=trained_models)

@app.route('/predict', methods=['POST'])
def make_prediction():
    data = request.json
    model_name = data.get('model')
    transaction_data = data.get('transaction_data', {})
    
    result = {
        'success': False,
        'prediction': None,
        'probability': 0,
        'message': ''
    }
    
    try:
        # Simulate prediction (integrate with Watson X in production)
        result['success'] = True
        result['prediction'] = np.random.choice(['Fraud', 'Legitimate'], p=[0.15, 0.85])
        result['probability'] = np.random.uniform(0.6, 0.99)
        result['message'] = 'Prediction completed successfully'
        
        session['last_prediction'] = f"{result['prediction']} ({result['probability']:.2%})"
        
    except Exception as e:
        result['message'] = str(e)
    
    return jsonify(result)

@app.route('/visualization')
def visualization_page():
    return render_template('visualization.html')

@app.route('/get_visualization_data')
def get_visualization_data():
    # Load sample data for visualization
    data = {
        'fraud_distribution': {
            'labels': ['Legitimate', 'Fraud'],
            'values': [99.85, 0.15]
        },
        'fraud_by_hour': {
            'hours': list(range(24)),
            'fraud_counts': [np.random.randint(0, 20) for _ in range(24)]
        },
        'fraud_by_amount': {
            'ranges': ['0-50', '50-100', '100-200', '200-500', '500+'],
            'counts': [10, 25, 45, 35, 20]
        },
        'top_mcc_codes': {
            'codes': ['5411', '5912', '5999', '5812', '5722'],
            'fraud_rates': [0.25, 0.18, 0.15, 0.12, 0.08]
        }
    }
    
    return jsonify(data)

@app.route('/api/watson/status')
def watson_status():
    # Check Watson X connection status
    status = {
        'connected': False,
        'message': 'Watson X integration pending',
        'dataset_available': os.path.exists('data/watsonx_final_dataset.csv')
    }
    
    return jsonify(status)

if __name__ == '__main__':
    app.run(debug=True, port=5000)