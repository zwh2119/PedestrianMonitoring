import cv2
import base_config
import numpy as np
import draw_service

genderProto = base_config.ROOT_DIR + "/gender_classification/models/gender_deploy.prototxt"
genderModel = base_config.ROOT_DIR + "/gender_classification/models/gender_net.caffemodel"
genderList = ['Male', 'Female']

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderNet = cv2.dnn.readNet(genderModel, genderProto)


def gender_service(frame, faceBoxes):
    padding = 20
    gender_res = []
    for faceBox in faceBoxes:
        faceBox = np.array(faceBox).astype(np.int32)
        left = max(0, faceBox[0] - padding)
        up = max(0, faceBox[1] - padding)
        right = min(faceBox[2] + padding, frame.shape[1] - 1)
        down = min(faceBox[3] + padding, frame.shape[0] - 1)
        face = frame[up:down, left:right]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender = genderList[genderPred[0].argmax()]
        gender_res.append(gender)

        labelGender = "{}".format("Gender : " + gender)
        cv2.putText(frame, labelGender, (left, up), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)

    return frame, gender_res

def test():
    img = cv2.imread('img.png')

    img, boxes = draw_service.get_face_box(img)
    img, gender_res = gender_service(img, boxes)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.imwrite('img_res.png',img)
