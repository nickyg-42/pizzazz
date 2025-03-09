from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import os
from audio_processor import process_audio_to_sheet_music

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file uploaded', 400
        
        file = request.files['file']
        if file.filename == '':
            return 'No file selected', 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the audio file
            output_pdf = process_audio_to_sheet_music(filepath)
            
            # Clean up the uploaded file
            os.remove(filepath)
            
            # Send the PDF file
            return send_file(output_pdf, as_attachment=True)
            
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)