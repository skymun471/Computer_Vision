from get_detected_img_mmdetec import get_detected_img
from mmdet.apis import init_detector, inference_detector
import cv2
import matplotlib.pyplot as plt

config_file = '/home/haneul/Desktop/DLCV/Computer_Vision/content/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = '/home/haneul/Desktop/DLCV/Computer_Vision/content/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')


demo = '/home/haneul/Desktop/DLCV/Computer_Vision/content/mmdetection/demo/demo.jpg'
beatles = '/home/haneul/Desktop/DLCV/Computer_Vision/content/data/beatles01.jpg'


img_array = cv2.imread(beatles)

detect_img = get_detected_img(
    model, img_array, score_threshold=0.5, is_print=True)
# detect된 img는 BGR임
detect_img_rgb = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 12))
plt.imshow(detect_img_rgb)
plt.show()
