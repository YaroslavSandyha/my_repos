import numpy as np
import cv2

# Включаем первую камеру
cap = cv2.VideoCapture(0)

ret, frame = cap.read()

# Записываем в файл
cv2.imwrite('cam.png', frame)
# Отключаем камеру
cap.release()
img = cv2.imread('cam.png', 0)
cv2.imwrite('cam2.png', img)
imag = cv2.imread('cam2.png', 1)
imag = cv2.line(imag,(0,0),(511,511),(0,255,0),7)
imag = cv2.rectangle(imag,(384,0),(510,128),(0,255,0),3)
cv2.imshow('image',imag)
cv2.waitKey(0)
cv2.destroyAllWindows()