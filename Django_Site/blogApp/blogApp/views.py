from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
import urllib.request
#----------------------------------------------------------------
from skimage.io import imread
import cv2 as cv
import numpy as np
import tkinter as tk
import png
import argparse
import os
from imutils import paths
from math import atan2, cos, sin, sqrt, pi
import imutils
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
from resizeimage import resizeimage
#----------------------------------------------------------------

#/home/atharvakango/Desktop/ProPy/Finger-Knuckle-based-Biometric-Identification-master

def croppedImg():
    img = cv.imread("/home/atharvakango/Desktop/ProPy/Finger-Knuckle-based-Biometric-Identification-master/test_data_model/downloaded.jpg")
    x1=1600
    y1=800
    x2=2400
    y2=1550
    crop_img = img[y1:y2, x1:x2]
    cover = cv.resize(crop_img,(int(160),int(180)))
    cv.imwrite('/home/atharvakango/Desktop/ProPy/Finger-Knuckle-based-Biometric-Identification-master/test_data_model/downloaded.jpg',cover)
    



def croppedImg2():
    img = cv.imread("/home/atharvakango/Desktop/ProPy/Finger-Knuckle-based-Biometric-Identification-master/Replica_Register/downloaded.jpg")
    x1=1600
    y1=800
    x2=2400
    y2=1550
    crop_img = img[y1:y2, x1:x2]
    cover = cv.resize(crop_img,(int(160),int(180)))
    cv.imwrite('/home/atharvakango/Desktop/ProPy/Finger-Knuckle-based-Biometric-Identification-master/Replica_Register/downloaded.jpg',cover)
    


def negativet():
    imagePaths = paths.list_images('/home/atharvakango/Desktop/ProPy/Finger-Knuckle-based-Biometric-Identification-master/test_data_model')

    for (i, imagePath) in enumerate(imagePaths):
        img = cv.imread(imagePath)
        label = imagePath.split(os.path.sep)[-1].split(".")[0]
        negative = 255 - img
        str1 = '/home/atharvakango/Desktop/ProPy/Finger-Knuckle-based-Biometric-Identification-master/test_data_model/downloaded.jpg'
        cv.imwrite(str1, negative)




def negative2():
    imagePaths = paths.list_images('/home/atharvakango/Desktop/ProPy/Finger-Knuckle-based-Biometric-Identification-master/Replica_Register')

    for (i, imagePath) in enumerate(imagePaths):
        img = cv.imread(imagePath)
        label = imagePath.split(os.path.sep)[-1].split(".")[0]
        negative = 255 - img
        str1 = '/home/atharvakango/Desktop/ProPy/Finger-Knuckle-based-Biometric-Identification-master/Replica_Register/downloaded.jpg'
        cv.imwrite(str1, negative)




def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)
    ## [visualization1]
    angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)

    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)

    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
    ## [visualization1]


def getOrientation(pts, img):
    ## [pca]
    # Construct a buffer used by the pca analysis
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)

    cntr = (int(mean[0, 0]), int(mean[0, 1]))
    cv.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0], cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0], cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
    drawAxis(img, cntr, p1, (0, 255, 0), 1)
    drawAxis(img, cntr, p2, (255, 255, 0), 5)
    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0]) 
    return angle


def pca():
    imagePaths =paths.list_images('/home/atharvakango/Desktop/ProPy/Finger-Knuckle-based-Biometric-Identification-master/test_data_model')

    for (i, imagePath) in enumerate(imagePaths):
        src = cv.imread(imagePath)
        label = imagePath.split(os.path.sep)[-1].split(".")[0]
        # Check if image is loaded successfully
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        _, bw = cv.threshold(gray, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        contours, h = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

        for i, c in enumerate(contours):
            # Calculate the area of each contour
            area = cv.contourArea(c);
            # Ignore contours that are too small or too large
            if area < 1e2 or 1e5 < area:
                continue

            # Draw each contour only for visualisation purposes
            cv.drawContours(src, contours, i, (0, 0, 255), 2);
            getOrientation(c, src)
        print(type(src))
        destination = '/home/atharvakango/Desktop/ProPy/Finger-Knuckle-based-Biometric-Identification-master/test_data_model/downloaded.jpg'
        cv.imwrite(destination, src)



def pca2():
    imagePaths =paths.list_images('/home/atharvakango/Desktop/ProPy/Finger-Knuckle-based-Biometric-Identification-master/Replica_Register')

    for (i, imagePath) in enumerate(imagePaths):
        src = cv.imread(imagePath)
        label = imagePath.split(os.path.sep)[-1].split(".")[0]
        # Check if image is loaded successfully
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        _, bw = cv.threshold(gray, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        contours, h = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

        for i, c in enumerate(contours):
            # Calculate the area of each contour
            area = cv.contourArea(c);
            # Ignore contours that are too small or too large
            if area < 1e2 or 1e5 < area:
                continue

            # Draw each contour only for visualisation purposes
            cv.drawContours(src, contours, i, (0, 0, 255), 2);
            getOrientation(c, src)
        print(type(src))
        destination = '/home/atharvakango/Desktop/ProPy/Finger-Knuckle-based-Biometric-Identification-master/Replica_Register/downloaded.jpg'
        cv.imwrite(destination, src)

def image_to_feature_vector(image, size=(32, 32)):
    return cv.resize(image, size).flatten()

def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hist = cv.calcHist([hsv], [0, 1, 2], None, bins,[0, 180, 0, 256, 0, 256])
    if imutils.is_cv2():
        hist = cv.normalize(hist)
    else:
        cv.normalize(hist, hist)
    return hist.flatten()

flg=0;

def knnclassifier_model(username):
    print("[INFO] describing images...")
    imagePaths = list(paths.list_images('/home/atharvakango/Desktop/ProPy/Finger-Knuckle-based-Biometric-Identification-master/converted_to_PCA'))
    # username = args["username"]
    rawImages = []
    features = []
    labels = []
    target = 0
    for (i, imagePath) in enumerate(imagePaths):
        image = cv.imread(imagePath)
        label = imagePath.split(os.path.sep)[-1].split(".")[0]
        pixels = image_to_feature_vector(image)
        hist = extract_color_histogram(image)
        rawImages.append(pixels)
        features.append(hist)
        labels.append(label)

        if i > 0 and i % 1000 == 0:
            print("[INFO] processed {}/{}".format(i, len(imagePaths)))
    rawImages = np.array(rawImages)
    features = np.array(features)
    labels = np.array(labels)
    print("[INFO] pixels matrix: {:.2f}MB".format(
        rawImages.nbytes / (1024 * 1000.0)))
    print("[INFO] features matrix: {:.2f}MB".format(
        features.nbytes / (1024 * 1000.0)))
    (trainRI, testRI, trainRL, testRL) = train_test_split(
        rawImages, labels, test_size=0.80, random_state=42)
    (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
        features, labels, test_size=0.20, random_state=42)

    print("[INFO] evaluating raw pixel accuracy...")

    kVals = range(1, 2, 2)
    accuracies = []
    for k in range(1, 2, 2):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(trainRI, trainRL)
        acc = model.score(testRI, testRL)
        accuracies.append(acc)

    i = int(np.argmax(accuracies))
    model = KNeighborsClassifier(n_neighbors=kVals[i])
    model.fit(trainRI, trainRL)
    predictions = model.predict(trainRI)

    print("EVALUTION ON TESTING DATA")
    print('------------------------------------------------------------------')

    imagePaths = list(paths.list_images('/home/atharvakango/Desktop/ProPy/Finger-Knuckle-based-Biometric-Identification-master/test_data_model'))
    print("image Path:" + str(imagePaths))
    for (i, imagePath) in enumerate(imagePaths):
        img = cv.imread(imagePath)
        label = imagePath.split(os.path.sep)[-1].split(".")[0]
        image = image_to_feature_vector(img)
        prediction = model.predict(image.reshape(1, -1))[0]
        image = image.reshape((32, 32, 3)).astype("uint8")
        print("prediction" + str(prediction))
        print("I think required image is: {}".format(prediction))
        splitted_username = str(prediction).split('_')
        if splitted_username[0] == username:
            flg=1
        else:
        	flg=0        
        return flg



#-----------------------------------------------------------------------------------------------------

def loginPage(request):
	if request.method == 'POST':
		th=request.POST.get('username')
		print (th)
		flink=request.POST.get('filename')
		print(flink)
		urllib.request.urlretrieve(flink,"/home/atharvakango/Desktop/ProPy/Finger-Knuckle-based-Biometric-Identification-master/test_data_model/downloaded.jpg")
		croppedImg()
		negativet()
		pca()
		flg=knnclassifier_model(th)
		if flg==1:
			return HttpResponse("User Verified!!")
		else:
			return HttpResponse( "Wrong Username or Password")

	return render(request, 'login.html')


def registerPage(request):
    if request.method == 'POST':
        th=request.POST.get('username')
        print (th)
        flink=request.POST.get('filename')
        print(flink)
        urllib.request.urlretrieve(flink,"/home/atharvakango/Desktop/ProPy/Finger-Knuckle-based-Biometric-Identification-master/Replica_Register/downloaded.jpg")
        replicateImage(th)
        return HttpResponse('Saved Successfully!!')
    return render(request,'register.html')


def testPage(request):
	return render(request, 'Web_App.html')



def replicateImage(username):
    i=0
    croppedImg2()
    negative2()
    pca2()
    img=cv.imread("/home/atharvakango/Desktop/ProPy/Finger-Knuckle-based-Biometric-Identification-master/Replica_Register/downloaded.jpg")
    for i in range(5):
        cv.imwrite('/home/atharvakango/Desktop/ProPy/Finger-Knuckle-based-Biometric-Identification-master/converted_to_PCA/'+str(username)+'_'+str(i)+'.jpg',img)