import csv
from matplotlib import pyplot as plt
import cv2
import base_config


def get_target_pic(video_path, idx, sign, car_nums):
    cap = cv2.VideoCapture(video_path)  # 返回一个capture对象
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # 设置要获取的帧号
    ret, frame = cap.read()  # read方法返回一个布尔值和一个视频帧。若帧读取成功，则返回True

    if ret == False:
        print("Error")
    else:
        dir_name = base_config.TARGET_DATA + "/" + "draw_car/"
        file_name = dir_name + "traffic_" + str(car_nums) + "_" + str(sign) + ".jpg"
        cv2.imwrite(file_name, frame)


def draw_bar():
    with open("nums.txt", 'r') as f:
        csv_reader = csv.reader(f)
        target_data = []
        for row in csv_reader:
            row = [int(i) for i in row]
            target_data.append(row)

    num_dict = {}
    for idx, item in enumerate(target_data[1]):
        if item not in num_dict.keys():
            num_dict[item] = []
        num_dict[item].append(idx)

    car_nums_set = set()
    # [8-26]

    for key, value in num_dict.items():
        car_nums_set.add(key)

    target_nums = [10, 13, 17, 20, 23]
    target_dict = {}

    for idx, item in enumerate(target_nums):
        target_dict[item] = num_dict[item]

    print(target_dict)

    # 采样30张
    video_path = base_config.ROOT_DIR + "/video_prepare/target_video/traffic.mp4"
    for key, value in target_dict.items():
        for i, item in enumerate(value):
            if i >= 30:
                break
            get_target_pic(video_path, item, i+1, key)

if __name__ == '__main__':
    draw_bar()
