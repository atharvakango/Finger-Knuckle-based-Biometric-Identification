# import the necessary packages
from typing import List, Union

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
from sklearn import metrics
from skimage import exposure


def image_to_feature_vector(image, size=(32, 32)):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, size).flatten()


def extract_color_histogram(image, bins=(8, 8, 8)):
    # extract a 3D color histogram from the HSV color space using
    # the supplied number of `bins` per channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])

    # handle normalizing the histogram if we are using OpenCV 2.4.X
    if imutils.is_cv2():
        hist = cv2.normalize(hist)

    # otherwise, perform "in place" normalization in OpenCV 3 (I
    # personally hate the way this is done
    else:
        cv2.normalize(hist, hist)

    # return the flattened histogram as the feature vector
    return hist.flatten()


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-t", "--test", required=True,
                help="path to input testdata")
# ap.add_argument("-t","--train",required=True,help="path to input train data")
ap.add_argument("-k", "--neighbors", type=int, default=1,
                help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
                help="# of jobs for k-NN distance (-1 uses all available cores)")
ap.add_argument("-u", "--username", default=-1,
                help="# Username of User")

args = vars(ap.parse_args())

# grab the list of images that we'll be describing
print("[INFO] describing images...")
imagePaths = list(paths.list_images(args["dataset"]))
username = args["username"]
# print('type------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>'+type(username))
# testPaths = list(paths.list_images(args["train"]))
# initialize the raw pixel intensities matrix, the features matrix,
# and labels list
rawImages = []
features = []
labels = []
target = 0
# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
    # load the image and extract the class label (assuming that our
    # path as the format: /path/to/dataset/{class}.{image_num}.jpg
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]
    # if label == '2_1':
    #     target=i
    # extract raw pixel intensity "features", followed by a color
    # histogram to characterize the color distribution of the pixels
    # in the image
    pixels = image_to_feature_vector(image)
    hist = extract_color_histogram(image)
    # update the raw images, features, and labels matricies,
    # respectively
    rawImages.append(pixels)
    features.append(hist)
    labels.append(label)

    # show an update every 1,000 images
    if i > 0 and i % 1000 == 0:
        print("[INFO] processed {}/{}".format(i, len(imagePaths)))

# for(i, testPath) in enumerate(testPaths):
#     image = cv2.imread(testPath)
#     label = testPath.split(os.path.sep)[-1].split(".")[0]
#     pixels = image_to_feature_vector(image)
#     hist = extract_color_histogram(image)


# show some information on the memory consumed by the raw images
# matrix and features matrix
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

imagePaths = list(paths.list_images(args["test"]))
print("image Path:"+str(imagePaths))
for (i, imagePath) in enumerate(imagePaths):
    img = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]

    # for i in list(map(int, np.random.randint(0, high=len(testRL), size=(5,)))):
    image =image_to_feature_vector(img)
    # cv2.imshow("Input",image)
    prediction = model.predict(image.reshape(1, -1))[0]
    # print(metrics.classification_report(testLabels, predictions))
    # print('--->>>>>>>>>>>>>>>>>' + str(len(image))+'>>>>>>>>>>>>>>>>>target'+str(target))
    image = image.reshape((32, 32, 3)).astype("uint8")
    # image = exposure.rescale_intensity(image,out_range=(0,255))
    # image = imutils.resize(image,width=32,inter=cv2.INTER_CUBIC)
    print("prediction"+str(prediction))
    print("I think required image is: {}".format(prediction))
    splitted_username = str(prediction).split('_')
    cv2.imshow("image:", image)
    if splitted_username[0]==username:
        print("User Verified")
    else:
        print("Username or password is wrong")
    cv2.waitKey(0)
    print('------------------------------------------------------------------')

#3,5,6,9