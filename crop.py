import cv2
from PIL import Image
from resizeimage import resizeimage



img = cv2.imread("C:/Users/guest123/Desktop/Software-Security-Using-Knucle-Print-master/test_data_model/downloaded.jpg")
x1=1600
y1=800

x2=2400
y2=1550

crop_img = img[y1:y2, x1:x2]
#crop_img = img[y:y+h, x:x+w]
#cv2.imshow("cropped", crop_img)
#cv2.imwrite('C:/Users/guest123/Desktop/Software-Security-Using-Knucle-Print-master/final.png',crop_img)
#resizeimage.resize_cover.validate(crop_img, [200, 100], validate=False)
cover = cv2.resize(crop_img,(int(160),int(180)))
#cover.save('test-image-cover.png', crop_img.format)
cv2.imshow("cropped", cover)
cv2.imwrite('C:/Users/guest123/Desktop/Software-Security-Using-Knucle-Print-master/test_data_model/downloaded_resize.jpg',cover)
cv2.waitKey(0)