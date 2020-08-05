from app import app
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import io
from PIL import Image
import base64
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
		flash('Document scan was successful')
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

		warped = transform(orig, screenCnt.reshape(4, 2) * ratio)

		img = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
		file_object = io.BytesIO()
		img= Image.fromarray(Helpers.resize(img,width=500))
		img.save(file_object, 'PNG')
		base64img = "data:image/png;base64,"+base64.b64encode(file_object.getvalue()).decode('ascii')

		return render_template('upload.html', image=base64img )
	else:
		flash('Allowed image types are -> png, jpg, jpeg')
		return redirect(request.url)

if __name__ == "__main__":
    app.run(debug=True)