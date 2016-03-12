__author__ = 'ypw'

import cv2
import numpy as np
from matplotlib import pylab as pl

face = cv2.imread("face/emo.mp40022.png", -1)
roi = face[60:670, 330:890]
cv2.imwrite("face/face.png", roi)
roi2 = roi[240:268, 190:270]
cv2.imwrite("face/mei.png", roi2)

roi3 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
print(roi3)
# pl.imshow(cv2.cvtColor(roi2, cv2.COLOR_BGR2RGB))
pl.imshow(cv2.cvtColor(roi3, cv2.COLOR_GRAY2RGB))
pl.show()

