import os
import cv2


from FaceDetection import FaceDetection
import base_config
import draw_service

image_root_path = base_config.TEST_DATA
image_target_path=base_config.TARGET_DATA
task_name='face_detection'
pic_name='target'
target_dir=image_target_path+'/'+task_name

args = {
    'net_type': 'mb_tiny_RFB_fd',
    'input_size': 1280,
    'threshold': 0.6,
    'candidate_size': 1000,
    'device': 'cuda:0',
}
face_detection = FaceDetection(args)
i=0
for root, dirs, files in os.walk(image_root_path):
    for file in files:
        i+=1
        file_path = image_root_path + '/' + file
        image = cv2.imread(file_path)
        boxes, labels, probs = face_detection(image)
        image=draw_service.draw_boxes(image,boxes)
        target_name=target_dir+'/'+pic_name+'_'+str(i)+'.jpg'
        print(target_name)
        cv2.imwrite(target_name,image)
