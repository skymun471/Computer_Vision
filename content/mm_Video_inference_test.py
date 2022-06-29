from do_detect_video_function import do_detect_video
from mmdet.apis import init_detector, inference_detector

config_file = '/home/haneul/Desktop/DLCV/Computer_Vision/content/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = '/home/haneul/Desktop/DLCV/Computer_Vision/content/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

do_detect_video(model, '/home/haneul/Desktop/DLCV/Computer_Vision/content/data/Jonh_Wick_small.mp4',
                '/home/haneul/Desktop/DLCV/Computer_Vision/content/data/Jonh_Wick_small_out.mp4', score_threshold=0.4, do_print=True)
