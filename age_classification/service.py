import cv2
import base_config
import numpy as np
import draw_service


ageProto = base_config.ROOT_DIR+"/age_classification/models/age_deploy.prototxt"
ageModel = base_config.ROOT_DIR+"/age_classification/models/age_net.caffemodel"
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageNet = cv2.dnn.readNet(ageModel, ageProto)

def age_service(frame,faceBoxes):
    padding = 20
    age_res = []
    for faceBox in faceBoxes:
        faceBox = np.array(faceBox).astype(np.int32)
        left = max(0, faceBox[0] - padding)
        up = max(0, faceBox[1] - padding)
        right = min(faceBox[2] + padding, frame.shape[1] - 1)
        down = min(faceBox[3] + padding, frame.shape[0] - 1)
        face = frame[up:down,left:right]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        ageNet.setInput(blob)
        agePred = ageNet.forward()
        age = ageList[agePred[0].argmax()]
        age_res.append(age)
        labelAge = "{}".format("Age : " + age + "Years")
        cv2.putText(frame, labelAge, (left,up), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)
    return frame,age_res

img = cv2.imread('img.png')
img, boxes = draw_service.get_face_box(img)
img,age_res = age_service(img, boxes)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.imwrite('img_res.png',img)
