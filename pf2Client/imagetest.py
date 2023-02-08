import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Create a black image
img = np.zeros((3*14,3*28,1), np.uint8)
# Draw a diagonal blue line with thickness of 5 px
cv.line(img,(28-1,14-1),(2*28+1,14-1),255,1)

cv.line(img,(2*28+1,14-1),(2*28+1,2*14+1),255,1)

cv.line(img,(2*28+1,2*14+1),(28-1,2*14+1),255,1)

cv.line(img,(28-1,2*14+1),(28-1,14-1),255,1)


plt.imshow(img)
plt.show()