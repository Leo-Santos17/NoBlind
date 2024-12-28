#!pip install ultralytics --break-system-packages
from ultralytics import YOLO
model = YOLO("yolo11m-pose.pt")

results = model("palestra.png")
results2 = model("palestra_2.png")
results3 = model("palestra_3.png")
results4 = model("palestra_4.png")

for result in results:
    boxer = result.boxes
    result.save(filename="result.jpg")
for result in results2:
    boxer = result.boxes
    result.save(filename="result_2.jpg")
for result in results3:
    boxer = result.boxes
    result.save(filename="result_3.jpg")
for result in results4:
    boxer = result.boxes
    result.save(filename="result_4.jpg")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = 'result.jpg'
image = mpimg.imread(img)
plt.imshow(image)
plt.axis('off')
plt.show()

img = 'palestra.png'

image = mpimg.imread(img)
plt.imshow(image)
plt.axis('off')
plt.show()

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = 'result_2.jpg'

image = mpimg.imread(img)
plt.imshow(image)
plt.axis('off')
plt.show()

img = 'palestra_2.png'

image = mpimg.imread(img)
plt.imshow(image)
plt.axis('off')
plt.show()