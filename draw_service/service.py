import cv2
import numpy as np
import base_config

faceProto = base_config.ROOT_DIR+"/draw_service/models/opencv_face_detector.pbtxt"
faceModel = base_config.ROOT_DIR+"/draw_service/models/opencv_face_detector_uint8.pb"
faceNet = cv2.dnn.readNet(faceModel, faceProto)
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

def draw_boxes(image,boxes):
    for i, box in enumerate(boxes):
        box=np.array(box,dtype=int)
        cv2.rectangle(image, pt1=(box[0], box[1]), pt2=(box[2], box[3]), color=(0, 255, 255), thickness=2)
    return image

def get_face_box(frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    faceBoxes = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    print(faceBoxes)
    return frame, faceBoxes