from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import cv2

data = []
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True,
	help="path to the training images")
args = vars(ap.parse_args())
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
image_scale = 1

for imagePath in paths.list_images(args["training"]):
	img = cv2.imread(imagePath)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
	    cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
	    roi_gray = gray[y:y+h, x:x+w]
	    roi_color = img[y:y+h, x:x+w]

	rezise_img = cv2.resize(roi_gray, (300,300))
	desc = LocalBinaryPatterns(24, 8)
	hist = desc.describe(rezise_img)
	data.append(hist)
# print(faces)
for x in data:
    with open('data.txt', 'a') as f:
        f.write(str(x)+',')
