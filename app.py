from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import os, uuid, threading, time
from werkzeug.utils import secure_filename
import pandas as pd
from analysis.pipeline import RNASeqPipeline
import tempfile
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for GitHub Pages integration

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# In-memory job storage (use Redis/DB for production)
jobs = {}

class JobStatus:
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"

@app.route('/')
def index():
    """Main upload interface"""
    return render_template('upload.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start analysis"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '' or not file.filename.endswith('.csv'):
            return jsonify({'error': 'Please upload a CSV file'}), 400
        
        # Generate job ID and save file
        job_id = str(uuid.uuid4())[:8]
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
        file.save(filepath)
        
        # Create job record
        jobs[job_id] = {
            'id': job_id,
            'status': JobStatus.PENDING,
            'filename': filename,
            'filepath': filepath,
            'created_at': datetime.utcnow().isoformat(),
            'message': 'File uploaded, analysis queued',
            'results': {}
        }
        
        # Start analysis in background
        threading.Thread(target=run_analysis, args=(job_id,)).start()
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'File uploaded successfully, analysis started',
            'status_url': f'/api/status/{job_id}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status/<job_id>')
def get_status(job_id):
    """Get job status"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    response = {
        'job_id': job_id,
        'status': job['status'],
        'message': job['message'],
        'created_at': job['created_at']
    }
    
    if job['status'] == JobStatus.COMPLETED:
        response['results'] = job['results']
    
    return jsonify(response)

@app.route('/api/download/<job_id>/<file_type>')
def download_result(job_id, file_type):
    """Download result files"""
    if job_id not in jobs or jobs[job_id]['status'] != JobStatus.COMPLETED:
        return jsonify({'error': 'Job not found or not completed'}), 404
    
    job = jobs[job_id]
    if file_type not in job['results']:
        return jsonify({'error': 'File not found'}), 404
    
    filepath = job['results'][file_type]['path']
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    else:
        return jsonify({'error': 'File no longer available'}), 404

@app.route('/api/jobs')
def list_jobs():
    """List recent jobs"""
    recent_jobs = []
    for job_id, job in sorted(jobs.items(), key=lambda x: x[1]['created_at'], reverse=True)[:10]:
        recent_jobs.append({
            'job_id': job_id,
            'filename': job['filename'],
            'status': job['status'],
            'created_at': job['created_at'],
            'message': job['message']
        })
    
    return jsonify({'jobs': recent_jobs})

def run_analysis(job_id):
    """Run RNA-seq analysis in background"""
    try:
        job = jobs[job_id]
        job['status'] = JobStatus.PROCESSING
        job['message'] = 'Running XGBoost analysis...'
        
        # Initialize pipeline
        pipeline = RNASeqPipeline()
        
        # Run analysis
        results = pipeline.analyze(job['filepath'], job_id)
        
        # Update job with results
        job['status'] = JobStatus.COMPLETED
        job['message'] = 'Analysis completed successfully'
        job['results'] = results
        job['completed_at'] = datetime.utcnow().isoformat()
        
    except Exception as e:
        job['status'] = JobStatus.FAILED
        job['message'] = f'Analysis failed: {str(e)}'
        print(f"Analysis failed for job {job_id}: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
