from flask import Flask, request, jsonify, render_template
from src.predict import predict_digit
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/" , methods=['GET','POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file uploaded', 400
        
        file = request.files['file']

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        digit = predict_digit(file_path)

        return render_template('index.html', prediction=digit, image_path=file_path)
    
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return '' , 204
        
    

if __name__ == '__main__':
    app.run(debug=True)