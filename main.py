import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import io
from PIL import Image

from Helpers import *

def orders(pts):
	rect = np.zeros((4, 2), dtype = "float32")
	s = pts.sum(axis = 1)

	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect

def transform(image, pts):
	rect = orders(pts)
	(tl, tr, br, bl) = rect

	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	return warped


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

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
		flash('Image successfully uploaded and displayed')
		filename = secure_filename(file.filename)
		
		filestr = request.files['file'].read()
		npimg = np.frombuffer(filestr, np.uint8)
		image = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
		ratio = image.shape[0] / 500.0
		orig = image.copy()
		image = Helpers.resize(image, height = 500)

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (5, 5), 0)
		edged = cv2.Canny(gray, 75, 200)

		cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		cnts = Helpers.grab_contours(cnts)
		cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

		for c in cnts:
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.02 * peri, True)
			if len(approx) == 4:
				screenCnt = approx
				break

		cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)

		file_object = io.BytesIO()
		img= Image.fromarray(image.astype('uint8'))
		img.save(file_object, 'PNG')
		base64img = "data:image/png;base64,"+b64encode(file_object.getvalue()).decode('ascii')

		warped = transform(orig, screenCnt.reshape(4, 2) * ratio)
		cv2.imwrite("static/output_{}".format(filename), Helpers.resize(warped, height = 500)) 

		return render_template('upload.html', filename="output_" + filename)
	else:
		flash('Allowed image types are -> png, jpg, jpeg')
		return redirect(request.url)

if __name__ == "__main__":
    app.run(debug=True)