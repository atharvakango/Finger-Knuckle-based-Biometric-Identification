import numpy as np
import cv2
import png
import argparse
import os
from imutils import paths

# cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
# ksize - size of gabor filter (n, n)
# sigma - standard deviation of the gaussian function
# theta - orientation of the normal to the parallel stripes
# lambda - wavelength of the sunusoidal factor
# gamma - spatial aspect ratio
# psi - phase offset
# ktype - type and range of values that each pixel in the gabor kernel can hold

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")


print("[INFO] describing images...")
args = vars(ap.parse_args())
imagePaths = list(paths.list_images(args["dataset"]))

g_kernel = cv2.getGaborKernel((21, 21),0.5, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
for (i, imagePath) in enumerate(imagePaths):
    img = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)
    # cv2.imshow('image', img)
    gray=cv2.imshow('filtered image', filtered_img)
    png.from_array(filtered_img,'L').save('C:/Users/guest123/Desktop/Software-Security-Using-Knucle-Print-master/converted_to_Gabor/'+str(label)+'_gabor.png')

    # h, w = g_kernel.shape[:2]
    # g_kernel = cv2.resize(g_kernel, (3*w, 3*h), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('gabor kernel (resized)', g_kernel)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()