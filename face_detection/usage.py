import os
import cv2


from FaceDetection import FaceDetection
import base_config

image_root_path = base_config.TEST_DATA

args = {
    'net_type': 'mb_tiny_RFB_fd',
    'input_size': 480,
    'threshold': 0.7,
    'candidate_size': 1500,
    'device': 'cuda:0',
}
face_detection = FaceDetection(args)
for root, dirs, files in os.walk(image_root_path):
    for file in files:
        file_path = image_root_path + '/' + file
        image = cv2.imread(file_path)
        boxes, labels, probs = face_detection(image)
        image_info = {
            'image_name': file,
            'image_mat': image.tolist(),
            'boxes': boxes.tolist(),
        }
