import csv
import cv2

from age_classification import age_service
from face_detection import detection_service
import base_config
import time
import random


# 5,10,15曲线都有5个横坐标
# 文件名称类似于time_5_360,time_5_720/time_10_1080/time_15_640这种
# 中缀为5/10/15的,对应同一条曲线

# time_5_360需要多张图片的数据,大概也就是120张

video_source_dir = base_config.ROOT_DIR + "/video_prepare/target_video"
video_name = "smallmanin"
video_target = video_source_dir + "/" + video_name+".mp4"
frame_target_dir = base_config.ROOT_DIR + "/video_prepare/frame_clip"

target_dir=base_config.TARGET_DATA+"/"+"draw_pic"

if __name__ == '__main__':

    # face number
    face = [5, 10, 15]
    # resolution
    # 480*240/640*360/720*480/1280*720/1920*1080
    reso = [240,360, 480, 720, 1080]
    target_reso=[(480,240),(640,360),(720,480),(1280,720),(1920,1080)]

    # 统计图片人数
    for face_num in face:
        for idx,res in enumerate(target_reso):
            with open(target_dir+"/"+"time_"+str(face_num)+"_"+str(res[1]),'w',newline="") as f:
                csv_writer = csv.writer(f)
                for i in range(1,151):
                    # print(i)
                    frame_name=frame_target_dir+"/"+video_name+"_"+str(i)+".jpg"
                    pic=cv2.imread(frame_name)
                    pic_resized=cv2.resize(pic,res,interpolation=cv2.INTER_CUBIC)
                    # cv2.imshow("win",pic_resized)
                    # cv2.waitKey(0)

                    time_start0=time.time()
                    boxes, labels, probs=detection_service(pic_resized)
                    time_end0=time.time()

                    # 随机抽face_num个锚框进行处理
                    # 如果不够face_num个就按原先的box处理
                    try:
                        target_boxes=random.sample(list(boxes),face_num)
                    except:
                        target_boxes=boxes

                    time_start1=time.time()
                    frame,age_res=age_service(pic_resized,target_boxes)
                    time_end1=time.time()

                    time_sum=time_end1-time_start1+time_end0-time_start0
                    csv_writer.writerow([time_sum])


