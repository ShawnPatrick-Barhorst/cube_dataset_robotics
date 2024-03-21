import cv2
import numpy as np
import onnx
import matplotlib.pyplot as plt

net=cv2.dnn.readNetFromONNX('opset_12_8.onnx')
#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

image = cv2.imread('test_image.jpg')

blob = cv2.dnn.blobFromImage(image, size=(128,128), scalefactor=1/255.0, mean=[0,0,0], swapRB=True, crop=False)
net.setInput(blob)
output = net.forward()
output = output.transpose((0, 2, 1))

print(output.shape)

dw, dh = (128,128)

bboxes = []
for prediction in output[0]:
    if prediction[4] > 0.75:
        x, y, w, h, confidence = prediction

        print(confidence)

        l = int((x - w / 2))
        r = int((x + w / 2))
        t = int((y - h / 2))
        b = int((y + h / 2))
    
        if l < 0:
            l = 0
        if r > dw - 1:
            r = dw - 1
        if t < 0:
            t = 0
        if b > dh - 1:
            b = dh - 1
        
        #bboxes.append([x1,y1,x2,y2])
        bboxes.append([l,r,t,b])

for box in bboxes:
    l, r, t, b = box
    cv2.rectangle(image, (l, t), (r, b), (0, 255, 0), 2)

cv2.imshow('detection', image)

while(True):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

#plt.imshow(image)
