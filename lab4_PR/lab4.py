import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img1 = cv.imread('train_img.jpg',1)# queryImage
img1 = cv.cvtColor(img1,cv.COLOR_BGR2RGB)
img2 = cv.imread('3.jpg',1) # trainImage
img2 = cv.cvtColor(img2,cv.COLOR_BGR2RGB)
img_replace = cv.imread('img_replace.jpg', 1)
img_replace = cv.cvtColor(img_replace,cv.COLOR_BGR2RGB)
# sift = cv.SIFT_create()
sift = cv.ORB_create(nfeatures=1000)
kp1, des1 = sift.detectAndCompute(img1,None)
kp_replace, des_replace = sift.detectAndCompute(img_replace,None)

cap = cv.VideoCapture('video.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
result = cv.VideoWriter('filename.avi',
                         cv.VideoWriter_fourcc(*'MJPG'),
                         10, size)

while(True):
    B = []
    ret, frame = cap.read()
    dim = (512, 756)
    img2 = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
    try:
        kp2, des2 = sift.detectAndCompute(img2, None)
        # bf = cv.BFMatcher()
        bf = cv.BFMatcher(cv.NORM_HAMMING)
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good.append(m)
        if len(good)>10:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
            h, w, d = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv.perspectiveTransform(pts,M)
            img_w = cv.warpPerspective(img_replace, M, ((img2.shape[1], img2.shape[0])))
            frame_white = np.zeros((img2.shape[0], img2.shape[1]), np.uint8)
            cv.fillPoly(frame_white, [np.int32(dst)], (255, 255, 255))
            frame_not = cv.bitwise_not(frame_white)
            img_copy = cv.bitwise_and(img2, img2, mask=frame_not)
            img2 = cv.bitwise_or(img_w, img_copy)
    except:
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img2,
                   'it looks like he\'s not here',
                   (50, 50),
                   font, 1,
                   (0, 255, 255),
                   2,
                   cv.LINE_4)
    cv.imshow('frame', img2)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()