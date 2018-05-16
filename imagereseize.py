from imutils import paths
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--resize", required=True,
	help="path to the resize images")
args = vars(ap.parse_args())

for imagePath in paths.list_images(args["resize"]):
	image = cv2.imread(imagePath)
	resized_image = cv2.resize(image, (0,0), fx=0.25, fy=0.25)
	cv2.imwrite(imagePath, resized_image)
