from mmdet.apis import show_result_pyplot
import mmcv
from mmdet.apis import init_detector, inference_detector
import torch
import cv2
import matplotlib.pyplot as plt
print(torch.__version__)

config_file = '/home/haneul/Desktop/DLCV/Computer_Vision/content/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = '/home/haneul/Desktop/DLCV/Computer_Vision/content/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

model = init_detector(config_file, checkpoint_file, device='cpu')

img = '/home/haneul/Desktop/DLCV/Computer_Vision/content/mmdetection/demo/demo.jpg'

img_arr = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
print(img_arr.shape)
plt.figure(figsize=(12, 12))
plt.imshow(img_arr)
# plt.show()

results = inference_detector(model, img)
print(type(results), len(results))

print(results[0].shape, results[1].shape, results[2].shape, results[3].shape)
# show_result_pyplot(model, img, results)

# print(model.__dict__)
# print(model.cfg.pretty_text)
