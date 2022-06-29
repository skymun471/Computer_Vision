from Video_faster_rcnn_function import do_detected_video
import cv2

cv_net = cv2.dnn.readNetFromTensorflow('/home/haneul/Desktop/DLCV/Computer_Vision/content/pretrained/faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb',
                                       '/home/haneul/Desktop/DLCV/Computer_Vision/content/pretrained/config_graph.pbtxt')

do_detected_video(cv_net, '/home/haneul/Desktop/DLCV/Computer_Vision/content/data/Jonh_Wick_small.mp4',
                  '/home/haneul/Desktop/DLCV/Computer_Vision/content/data/Jonh_Wick_small_out_cv.mp4', 0.2, False)
