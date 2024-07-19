import cv2
import base_config

video_source_dir = base_config.ROOT_DIR + "/video_prepare/target_video"
video_name = "traffic"
video_target = video_source_dir + "/" + video_name+".mp4"
frame_target_dir = base_config.ROOT_DIR + "/video_prepare/frame_clip"

if __name__ == '__main__':
    capture = cv2.VideoCapture(video_target)
    if capture.isOpened():
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        i = 0
        while True:
            success, frame = capture.read()
            if success:
                i+=3
                frame_name = video_name+"_"+str(i//3)+".jpg"
                # cv2.imshow("win",frame)
                # cv2.waitKey(0)
                cv2.imwrite(frame_target_dir+"/"+frame_name,frame)
            if i>=450:
                break