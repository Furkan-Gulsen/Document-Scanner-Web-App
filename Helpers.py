import cv2

class Helpers:
	def __init__(self):
		pass

	def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
	    dim = None
	    (h, w) = image.shape[:2]
	    if width is None and height is None:
	        return image
	    if width is None:
	        r = height / float(h)
	        dim = (int(w * r), height)
	    else:
	        r = width / float(w)
	        dim = (width, int(h * r))
	    resized = cv2.resize(image, dim, interpolation=inter)

	    return resized

	def grab_contours(cnts):
		if len(cnts) == 2:
			cnts = cnts[0]
		elif len(cnts) == 3:
			cnts = cnts[1]
		else:
			raise Exception('The length of the contour must be 2 or 3.')
		return cnts