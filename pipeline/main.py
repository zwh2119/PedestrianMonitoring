import base_config
import os
import cv2

from age_classification import age_service
from face_detection import detection_service
from gender_classification import gender_service
import draw_service
import time


def test_age():
    image_root_path = base_config.TEST_DATA
    image_target_path = base_config.TARGET_DATA
    task_name_age = 'age_classification'
    pic_name = 'target'
    target_dir_age = image_target_path + '/' + task_name_age
    i = 0
    time_sum_det = 0
    time_sum_age = 0

    for root, dirs, files in os.walk(image_root_path):
        for file in files:
            i += 1
            file_path = image_root_path + '/' + file
            image = cv2.imread(file_path)

            time_start_det = time.time()
            boxes, labels, probs = detection_service(image)
            time_end_det = time.time()

            time_start_age = time.time()
            frame_age, age_res = age_service(image, boxes)
            time_end_age = time.time()

            time_sum_det += time_end_det - time_start_det
            time_sum_age += time_end_age - time_start_age

            target_name_age = target_dir_age + '/' + pic_name + 'age' + '_' + str(i) + '.jpg'
            cv2.imwrite(target_name_age, frame_age)

    print(time_sum_det)
    print(time_sum_age)

def test_gender():
    image_root_path = base_config.TEST_DATA
    image_target_path = base_config.TARGET_DATA
    task_name_gender = 'gender_classification'
    pic_name = 'target'
    target_dir_gender = image_target_path + '/' + task_name_gender
    i = 0
    time_sum_det = 0
    time_sum_gender = 0

    for root, dirs, files in os.walk(image_root_path):
        for file in files:
            i += 1
            file_path = image_root_path + '/' + file
            image = cv2.imread(file_path)

            time_start_det = time.time()
            boxes, labels, probs = detection_service(image)
            time_end_det = time.time()

            time_start_gender= time.time()
            frame_gender, gender_res = gender_service(image, boxes)
            time_end_gender = time.time()

            time_sum_det += time_end_det - time_start_det
            time_sum_gender += time_end_gender - time_start_gender

            target_name_gender = target_dir_gender + '/' + pic_name + 'gender' + '_' + str(i) + '.jpg'
            cv2.imwrite(target_name_gender, frame_gender)

    print(time_sum_det)
    print(time_sum_gender)

if __name__ == '__main__':
    test_age()
    # test_gender()
    # print(cv2.cuda.getCudaEnabledDeviceCount())
    # print(cv2.getBuildInformation())
