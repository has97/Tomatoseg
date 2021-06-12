import os

from tensorflow.python.keras.preprocessing.image import save_img
from app import app
import urllib.request
import main
import cv2
from main import getPrediction
# from tensorflow.keras.preprocessing import load_img,save_img
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
	    flash('No file part')
	    return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
	    flash('No image selected for uploading')
	    return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img,label= getPrediction(filename)
        flash(label)
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename), img)
        return render_template('upload.html', filename=filename)

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run(debug=True)