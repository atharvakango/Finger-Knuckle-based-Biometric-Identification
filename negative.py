from skimage.io import imread
import cv2
import numpy as np
import png
import argparse
import os
from imutils import paths

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,help="path to input dataset")

args = vars(ap.parse_args())
imagePaths = list(paths.list_images(args["dataset"]))



for (i, imagePath) in enumerate(imagePaths):
	img = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-1].split(".")[0]
	negative = 255 - img
	#str=label+'_pca'
	#png.from_array(negative,'L').save('C:/Users/guest123/Desktop/Software-Security-Using-Knucle-Print-master/Negative_converted/'+str(label)+'_Negative.png')
	str1='C:/Users/guest123/Desktop/Software-Security-Using-Knucle-Print-master/Negative_converted/'+str(label)+'.jpg'
	cv2.imwrite(str1, negative)
    #cv2.imwrite('C:/Users/guest123/Desktop/Software-Security-Using-Knucle-Print-master/Negative_converted/'+str, negative)

    
    
    














cv2.imshow('as',negative)
cv2.waitKey(0)