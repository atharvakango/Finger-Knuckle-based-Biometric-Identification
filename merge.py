from skimage.io import imread
import cv2 as cv
import numpy as np
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
import tkinter as tk

def croppedImg():
    img = cv2.imread("C:/Users/guest123/Desktop/Software-Security-Using-Knucle-Print-master/test_data_model/downloaded.jpg")
    x1=1600
    y1=800
    x2=2400
    y2=1550
    crop_img = img[y1:y2, x1:x2]
    cover = cv2.resize(crop_img,(int(160),int(180)))
    cv2.imwrite('C:/Users/guest123/Desktop/Software-Security-Using-Knucle-Print-master/test_data_model/downloaded.jpg',cover)
    


def negativet():
    imagePaths = paths.list_images('C:/Users/guest123/Desktop/Software-Security-Using-Knucle-Print-master/test_data_model')

    for (i, imagePath) in enumerate(imagePaths):
        img = cv.imread(imagePath)
        label = imagePath.split(os.path.sep)[-1].split(".")[0]
        negative = 255 - img
        str1 = 'C:/Users/guest123/Desktop/Software-Security-Using-Knucle-Print-master/test_data_model/downloaded.jpg'
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

    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)

    # Store the center of the object
    cntr = (int(mean[0, 0]), int(mean[0, 1]))
    ## [pca]

    ## [visualization]
    # Draw the principal components
    cv.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (
    cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0], cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    p2 = (
    cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0], cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
    drawAxis(img, cntr, p1, (0, 255, 0), 1)
    drawAxis(img, cntr, p2, (255, 255, 0), 5)

    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians
    ## [visualization]

    return angle


def pca():
    imagePaths =paths.list_images('C:/Users/guest123/Desktop/Software-Security-Using-Knucle-Print-master/test_data_model')

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
            # Find the orientation of each shape
            getOrientation(c, src)
        print(type(src))
        destination = 'C:/Users/guest123/Desktop/Software-Security-Using-Knucle-Print-master/test_data_model/downloaded.jpg'
        cv.imwrite(destination, src)

def image_to_feature_vector(image, size=(32, 32)):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv.resize(image, size).flatten()

def extract_color_histogram(image, bins=(8, 8, 8)):
    # extract a 3D color histogram from the HSV color space using
    # the supplied number of `bins` per channel
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hist = cv.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])

    # handle normalizing the histogram if we are using OpenCV 2.4.X
    if imutils.is_cv2():
        hist = cv.normalize(hist)

    # otherwise, perform "in place" normalization in OpenCV 3 (I
    # personally hate the way this is done
    else:
        cv.normalize(hist, hist)

    # return the flattened histogram as the feature vector
    return hist.flatten()


def knnclassifier_model():
    ap = argparse.ArgumentParser()
    ap.add_argument("-u", "--username", default=-1,
                    help="# Username of User")

    args = vars(ap.parse_args())

    # grab the list of images that we'll be describing
    print("[INFO] describing images...")
    imagePaths = list(paths.list_images('/home/sagar/AK_PCA'))
    username = args["username"]
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

        # show an update every 1,000 images
        if i > 0 and i % 1000 == 0:
            print("[INFO] processed {}/{}".format(i, len(imagePaths)))
    rawImages = np.array(rawImages)
    features = np.array(features)
    labels = np.array(labels)
    print("[INFO] pixels matrix: {:.2f}MB".format(
        rawImages.nbytes / (1024 * 1000.0)))
    print("[INFO] features matrix: {:.2f}MB".format(
        features.nbytes / (1024 * 1000.0)))
    # print("labels:"+str(labels))
    # partition the data into training and testing splits, using 75%
    # of the data for training and the remaining 25% for testing
    (trainRI, testRI, trainRL, testRL) = train_test_split(
        rawImages, labels, test_size=0.80, random_state=42)
    (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
        features, labels, test_size=0.20, random_state=42)

    # train and evaluate a k-NN classifer on the raw pixel intensities
    print("[INFO] evaluating raw pixel accuracy...")

    kVals = range(1, 2, 2)
    accuracies = []
    for k in range(1, 2, 2):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(trainRI, trainRL)
        acc = model.score(testRI, testRL)
        # print("k=%d, accuracy=%.2f%%" % (k, acc * 100))
        accuracies.append(acc)

    i = int(np.argmax(accuracies))
    # print("k=%d achieved highest accuracy of %.2f%% on validation data " % (kVals[i], accuracies[i] * 100))
    model = KNeighborsClassifier(n_neighbors=kVals[i])
    model.fit(trainRI, trainRL)
    predictions = model.predict(trainRI)

    print("EVALUTION ON TESTING DATA")
    # print(metrics.classification_report(testLabels, predictions))
    # print(str(trainRL))
    print('------------------------------------------------------------------')

    imagePaths = list(paths.list_images('/home/sagar/AK_TEST/'))
    print("image Path:" + str(imagePaths))
    for (i, imagePath) in enumerate(imagePaths):
        img = cv.imread(imagePath)
        label = imagePath.split(os.path.sep)[-1].split(".")[0]

        # for i in list(map(int, np.random.randint(0, high=len(testRL), size=(5,)))):
        image = image_to_feature_vector(img)
        # cv2.imshow("Input",image)
        prediction = model.predict(image.reshape(1, -1))[0]
        # print(metrics.classification_report(testLabels, predictions))
        # print('--->>>>>>>>>>>>>>>>>' + str(len(image))+'>>>>>>>>>>>>>>>>>target'+str(target))
        image = image.reshape((32, 32, 3)).astype("uint8")
        # image = exposure.rescale_intensity(image,out_range=(0,255))
        # image = imutils.resize(image,width=32,inter=cv2.INTER_CUBIC)
        print("prediction" + str(prediction))
        print("I think required image is: {}".format(prediction))
        splitted_username = str(prediction).split('_')
        #  cv2.imshow("image:", image)
        main = tk.Tk()
        main.geometry('500x300')
        main.config(bg='light blue')
        main.title('Login Information')
        # Tk.Frame(main, width=100, height=100)
        if splitted_username[0] == username:
            tk.Label(main,
                     text="User Verified",
                     fg="dark green",
                     bg="light green",
                     font=("Helvetica 16 bold italic", 50)).pack()
        else:
            tk.Label(main,
                     text="Wrong Username or Password",
                     fg="dark green",
                     bg="light green",
                     font="Helvetica 16 bold italic").pack()

        main.mainloop()
        # cv2.waitKey(0)
        print('------------------------------------------------------------------')

    # 3,5,6,9

croppedImg()
negativet()
pca()
#knnclassifier_model()