from faster_rcnn_img_function import get_detected_img
import cv2
import matplotlib.pyplot as plt

# image 로드
img = cv2.imread('/home/haneul/Desktop/DLCV/Computer_Vision/content/data/baseball01.jpg')
print('image shape:', img.shape)

# tensorflow inference 모델 로딩
cv_net = cv2.dnn.readNetFromTensorflow('/home/haneul/Desktop/DLCV/Computer_Vision/content/pretrained/faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb',
                                       '/home/haneul/Desktop/DLCV/Computer_Vision/content/pretrained/config_graph.pbtxt')
# Object Detetion 수행 후 시각화
draw_img = get_detected_img(
    cv_net, img, score_threshold=0.5, use_copied_array=True, is_print=True)

img_rgb = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 12))
plt.imshow(img_rgb)
plt.show()
