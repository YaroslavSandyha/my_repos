import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time


def box(output,papka,detector,painted = False,normType = cv.NORM_L2):
    f = open(output+'\\1_true_{q}.txt'.format(q=papka), 'w')
    f2 = open(output+'\\1_localization_{q}.txt'.format(q=papka), 'w')
    f3 = open(output+'\\1_time_{q}.txt'.format(q=papka), 'w')
    f4 = open(output+'\\2_true_{q}.txt'.format(q=papka), 'w')
    f5 = open(output+'\\2_localization_{q}.txt'.format(q=papka), 'w')
    f6 = open(output+'\\2_time_{q}.txt'.format(q=papka), 'w')
    img1 = cv.imread(papka+'\\true.jpg', 1)
    img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
    kp1, des1 = detector.detectAndCompute(img1, None)
    bf = cv.BFMatcher(normType)
    norm = len(bf.knnMatch(des1, des1, k=2))
    for i in range (1, 121):
        start_time  = time.time()
        img2 = cv.imread(papka+"\\{id}.jpg".format(id=i),1)
        img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

        kp2, des2 = detector.detectAndCompute(img2, None)
        good_point_two = []
        good_point = []
        matches = bf.knnMatch(des1,des2, k=2)
        if (len(matches[0])!=1):
            for m, n in matches:
                if m.distance < 0.89* n.distance:
                    good_point.append([m])
                    good_point_two.append([(pow(m.distance,1/2))])
            if(i<=100):
                f.write('img' + str(i) + ' ' + str(len(good_point))+' / '+ str(norm) + '\n')
                f2.write('img' + str(i) + ' ' + str( np.mean(good_point_two)) + '\n')
                f3.write('img' + str(i) + "--- %s seconds ---" % (time.time() - start_time) + '\n')

            elif(i>100):
                f4.write('img' + str(i) + ' ' + str(len(good_point))+ '/'+ str(norm) + '\n')
                f5.write('img' + str(i) + ' ' + str(np.mean(good_point_two)) + '\n')
                f6.write('img' + str(i) + "--- %s seconds ---" % (time.time() - start_time) + '\n')
            if painted:
                img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good_point, None, flags=2)
                plt.imshow(img3),plt.show()
        else:
            f.write('img' + str(i) + ' no matches' +'\n')
            f2.write('img' + str(i) + ' no matches' + '\n')
            f3.write('img' + str(i) + ' no matches' +'\n')
            f4.write('img' + str(i) + ' no matches' + '\n')
            f5.write('img' + str(i) + ' no matches' +'\n')
            f6.write('img' + str(i) + ' no matches' +'\n')
        print(i)


box("ORB","Kharchenko_A",cv.ORB_create(), painted = False,normType = cv.NORM_HAMMING)