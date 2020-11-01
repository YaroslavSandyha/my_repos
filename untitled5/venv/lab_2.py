import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
from sklearn import svm
from sklearn.decomposition import PCA
#from sklearn.cross_validation import train_test_split
# def ORB():
#     training = [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#                  1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1,
#                  1, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0,
#                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     return training
# def SIFT():
#     training= [1,0,1,1,1,1,0,0,0,0,
#                 1,1,0,1,1,1,1,1,0,0,
#                 1,1,1,0,1,0,0,0,0,0,
#                 0,0,0,0,1,0,0,0,0,0,
#                 0,0,0,1,1,1,0,0,0,1,
#                 0,1,1,1,1,1,1,1,1,1,
#                 1,1,0,1,1,1,0,1,1,1,
#                 1,1,0,0,0,1,0,0,0,1]
#     return training
# def AKAZE():
#     training = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#                 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0,
#                 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0,
#                 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1,
#                 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
#                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
#     return training
# def box(output,papka,papka2, detector,painted = False,normType = cv.NORM_L2):
#     f = open(output+'\\1_true_{q}.txt'.format(q=papka), 'w')
#     f2 = open(output+'\\1_localization_{q}.txt'.format(q=papka), 'w')
#     f3 = open(output+'\\1_time_{q}.txt'.format(q=papka), 'w')
#     f4 = open(output+'\\2_true_{q}.txt'.format(q=papka), 'w')
#     f5 = open(output+'\\2_localization_{q}.txt'.format(q=papka), 'w')
#     f6 = open(output+'\\2_time_{q}.txt'.format(q=papka), 'w')
#     img1 = cv.imread(papka+'\\true.jpg', 1)
#     img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
#     kp1, des1 = detector.detectAndCompute(img1, None)
#     bf = cv.BFMatcher(normType)
#     norm = len(bf.knnMatch(des1, des1, k=2))
#     pca = PCA(n_components=1)
#     desi =[]
#     support = svm.SVC()
#     for i in range (1, 47):
#         start_time  = time.time()
#         img2 = cv.imread(papka2+"\\{id}.jpg".format(id=i),1)
#         img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
#
#         kp2, des2 = detector.detectAndCompute(img2, None)
#         X_pca = pca.fit_transform(des2.T).reshape(1, -1)
#         desi.append(X_pca[0])
#         print (i)
#     support.fit(desi, training)
#     pred = support.predict(desi)
#     print (pred)
#     desi = []
#     for i in range ( 40, 121):
#         start_time = time.time()
#         img2 = cv.imread(papka + "\\{id}.jpg".format(id=i), 1)
#         img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
#         kp2, des2 = detector.detectAndCompute(img2, None)
#         X_pca = pca.fit_transform(des2.T).reshape(1, -1)
#         desi.append(X_pca[0])
#         print(i)
#     pred = support.predict(desi)
#     print(pred)
#
# box("AKAZE","Sandyha_Y","testing",cv.ORB_create(), painted = False,normType = cv.NORM_HAMMING)
cap = cv.VideoCapture(0)
ret, frame = cap.read()
dim = (512, 756)
frame2 = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
cv.imshow('frame', frame2)

#cap.release()
#cv.destroyAllWindows()