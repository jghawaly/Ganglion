import cv2
import os


img_dir = "C:\\Users\\james\\CODE\\Ganglion\\ganglion\\data\\"
imgs = []

for filename in os.listdir(img_dir):
    if filename.endswith(".JPEG"):
        img = cv2.imread(os.path.join(img_dir, filename), 0)
        img = cv2.resize(img, (64, 64), cv2.INTER_AREA)
        cv2.imshow("myimg", img)
        cv2.waitKey(0)

