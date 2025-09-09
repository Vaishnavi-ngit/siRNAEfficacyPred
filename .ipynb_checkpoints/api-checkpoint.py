from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os

# Import your actual inference function
from src.infer import main  # Assumes src/data/main.py contains a `main()` function

app = Flask(__name__)

# Define the upload folder relative to the script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'src', 'data', 'input')

# Create the upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Set a maximum file size limit (16 MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

@app.route('/', methods=['GET'])
def home():
    return "API is running. Use POST /upload to upload siRNA and mRNA FASTA files."

@app.route('/favicon.ico', methods=['GET'])
def favicon():
    return '', 204  # No Content

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        if 'siRNA' not in request.files or 'mRNA' not in request.files:
            return jsonify({"error": "Both siRNA and mRNA files are required"}), 400

        siRNA_file = request.files['siRNA']
        mRNA_file = request.files['mRNA']

        if siRNA_file.filename == '' or mRNA_file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        siRNA_filename = secure_filename(siRNA_file.filename)
        mRNA_filename = secure_filename(mRNA_file.filename)

        if not siRNA_filename.endswith(('.fasta', '.fa')) or not mRNA_filename.endswith(('.fasta', '.fa')):
            return jsonify({"error": "Files must be in FASTA format"}), 400

        # Save the files
        siRNA_path = os.path.join(UPLOAD_FOLDER, siRNA_filename)
        mRNA_path = os.path.join(UPLOAD_FOLDER, mRNA_filename)

        siRNA_file.save(siRNA_path)
        mRNA_file.save(mRNA_path)

        # Run inference after saving the files
        result = main()  # Replace this with main(siRNA_path, mRNA_path) if your function takes inputs

        return jsonify({
            "message": "Inference completed successfully",
            "output": result
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
