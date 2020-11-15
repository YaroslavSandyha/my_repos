import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
from sklearn import svm
from sklearn.decomposition import PCA

def box(output,papka, detector,normType = cv.NORM_L2):
    bf = cv.BFMatcher(normType)
    pca = PCA(n_components=1)
    desi = []
    support = svm.SVC()
    training = []
    test = []
    with open(papka+"\\training\\training.txt") as file:
        training = [int(x) for x in file.readline().split()];
    with open(papka+"\\test\\test.txt") as file:
        test = [int(x) for x in file.readline().split()];
    starttime = time.time()
    for i in range(1, len(training)+1):
         img2 = cv.imread(papka+"\\training\\{id}.jpg".format(id=i),1)
         img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
         kp2, des2 = detector.detectAndCompute(img2, None)
         X_pca = pca.fit_transform(des2.T).reshape(1, -1)
         desi.append(X_pca[0])
         print(i)
    support.fit(desi, training)
    desi = []
    for i in range( 1, len(test) + 1):
        img2 = cv.imread(papka + "\\test\\{id}.jpg".format(id=i), 1)
        img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
        kp2, des2 = detector.detectAndCompute(img2, None)
        X_pca = pca.fit_transform(des2.T).reshape(1, -1)
        desi.append(X_pca[0])
    pred = support.predict(desi)
    print(pred)
    endtime = time.time()
    test_true_realy1 = 0
    test_true1 = 0
    test_true0 = 0
    test_true_realy0 = 0
    for i in range(0,len(test)):
        if test[i] == 0:
            if test[i] != pred[i]:
                test_true0+=1
            test_true_realy0 +=1
        else:
            if test[i] != pred[i]:
                test_true1+=1
            test_true_realy1 +=1
    print(output)
    print("false negative",test_true0,'\\',test_true_realy0)
    print("false positive",test_true1, '\\', test_true_realy1)
    print(test_true0 + test_true1, '\\', test_true_realy0+test_true_realy1)
    print("Time: ",endtime - starttime)

    cap = cv.VideoCapture(papka + '\\video2.mp4')
    img1 = cv.imread(papka + "\\training\\1.jpg", 1)
    des1 = detector.detectAndCompute(img1, None)
    while (True):
        B = []
        ret, frame = cap.read()
        dim = (512, 756)
        frame2 = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
        try:
            kp2, des2 = detector.detectAndCompute(frame2, None)
            desq = pca.fit_transform(des2.T).reshape(1, -1)
            B.append(desq[0])
            y = support.predict(B)
            if y[0] == 1:
                font = cv.FONT_HERSHEY_SIMPLEX
                cv.putText(frame2,
                           'Na meste',
                           (50, 50),
                           font, 1,
                           (0, 255, 255),
                           2,
                           cv.LINE_4)
        except:
            pass
        cv.imshow('frame', frame2)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()


box("SIFT","Sandyha_Y",cv.SIFT_create(),normType = cv.NORM_L2)