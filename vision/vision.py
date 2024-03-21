import cv2
import numpy as np
import onnx
import time


MIN_CONFIDENCE = 0.7
net = cv2.dnn.readNet('opset_12_8.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def process(img):
    img = cv2.resize(img, (128, 128), interpolation = cv2.INTER_LINEAR)
    bilateral = cv2.bilateralFilter(img, 9, 355, 355)
    edge = cv2.Canny(img, 50, 500)
    output  = cv2.merge((edge, edge, edge))
    return output

cap = cv2.VideoCapture(-2)

dw, dh = (128, 128)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    #processing
    image = process(frame)

    #input[0:128, 0:128, 0:3] = image
    #dnn
    blob = cv2.dnn.blobFromImage(image, size=(128,128), scalefactor=1/255.0, mean=[0,0,0], swapRB=True, crop=False)
    net.setInput(blob)
    output = net.forward()
    output = output.transpose((0, 2, 1))
    #print(output[0][0].shape)

    bboxes = []
    for prediction in output[0]:
        if prediction[4] > MIN_CONFIDENCE:
            x, y, w, h, confidence = prediction

            l = int((x-w / 2))
            r = int((x+w / 2))
            t = int((y-h / 2))
            b = int((y+h / 2))

            if l < 0:
                l=0
            if r > dw - 1:
                r = dw - 1
            if t < 0:
                t = 0
            if b > dh - 1:
                b = dh - 1

            bboxes.append([l, r, t, b, w, h])

    for box in bboxes:
        l, r, t, b, w, h = box
        cv2.rectangle(image, (l, t), (r, b), (0,255,0), 2)
        cv2.putText(image, str(confidence), (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        centroid_x = int(l + w // 2)
        centroid_y = int(t + h // 2)
        print((centroid_x, centroid_y))
        cv2.circle(image, (centroid_x, centroid_y), 3, (0, 0, 255), -1)

    # Display the resulting frame
    if frame.size != 0:

        image = cv2.resize(image, (512, 512))
        cv2.imshow('frame', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.5)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
