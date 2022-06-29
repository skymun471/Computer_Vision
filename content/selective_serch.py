from turtle import color
import numpy as np
import selectivesearch
import cv2
import matplotlib.pyplot as plt
import os
# %matplotlib inline

# 오드리헵번 이미지를 cv2로 로드하고 matplotlib으로 시각화
# img = cv2.imread('./data/audrey01.jpg')
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# print('img shape:', img.shape)

# plt.figure(figsize=(8, 8))
# plt.imshow(img_rgb)
# plt.show()

# region proposal 정보를 반환한다.
# _, regions = selectivesearch.selective_search(
#     img_rgb, scale=100, min_size=2000)
# print(type(regions), len(regions))

#  rect 정보만 출력해서 보기
#  cand는 딕셔너리 인덱스의 변수 값을 반환 한다.
# cand_rect = [cand['rect'] for cand in regions]
# print(cand_rect)

# bounding_box 시각화하기
green_rgb = (125, 255, 51)
# for rect in cand_rect:
#     left = rect[0]
#     top = rect[1]
#     right = left + rect[2]
#     buttom = top + rect[3]

#     img_rgb = cv2.rectangle(img_rgb, (left, top),
#                             (right, buttom), color=green_rgb, thickness=2)


# selectiveserche region proposal 값을 cv2.rectangel을 통해 표시 이미지
# plt.figure(figsize=(8, 8))
# plt.imshow(img_rgb)
# plt.show()

# iou 구하기


def compute_iou(proposal_box, gt_box):
    # calculate intersection areas
    x1 = np.maximum(proposal_box[0], gt_box[0])
    y1 = np.maximum(proposal_box[1], gt_box[1])
    x2 = np.minimum(proposal_box[2], gt_box[2])
    y2 = np.minimum(proposal_box[3], gt_box[3])

    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

    proposal_box_area = (
        proposal_box[2] - proposal_box[0]) * (proposal_box[3] - proposal_box[1])
    gt_box_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    union = proposal_box_area + gt_box_area - intersection

    iou = intersection / union
    return iou


# 실제 box(Ground Truth)의 좌표를 아래와 같다고 가정.
gt_box = [60, 15, 320, 420]
red = (255, 0, 0)
# img_rgb = cv2.rectangle(
#     img_rgb, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), color=red, thickness=2)

# gt_box 가정 표시 이미지
# plt.figure(figsize=(8, 8))
# plt.imshow(img_rgb)
# plt.show()

# index 와 iou 출력
# for index, proposal_box in enumerate(cand_rect):
#     proposal_box = list(proposal_box)
#     proposal_box[2] += proposal_box[0]
#     proposal_box[3] += proposal_box[1]

#     iou = compute_iou(proposal_box, gt_box)
#     # print('index:', index, 'iou:', iou)

# cand_rect = [cand['rect'] for cand in regions if cand['size'] > 5000]
# cand_rect.sort()
# print(cand_rect)


img = cv2.imread(
    '/Users/munhaneul/Haneul/develop/Computer_vision/practice/content/data/audrey01.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print('img shape:', img.shape)
_, regions = selectivesearch.selective_search(
    img_rgb, scale=100, min_size=2000)
cand_rects = [cand['rect'] for cand in regions if cand['size'] > 3000]
img_rgb = cv2.rectangle(
    img_rgb, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), color=red, thickness=2)

for index, cand_box in enumerate(cand_rects):
    cand_box = list(cand_box)
    cand_box[2] += cand_box[0]
    cand_box[3] += cand_box[1]

    iou = compute_iou(cand_box, gt_box)

    if iou > 0.5:
        print('index:', index, 'iou:', iou, 'rectangle:',
              (cand_box[0], cand_box[1], cand_box[2], cand_box[3]))
        cv2.rectangle(img_rgb, (cand_box[0], cand_box[1]),
                      (cand_box[2], cand_box[3]), color=green_rgb, thickness=1)
        text = "{}: {:.2f}".format(index, iou)
        cv2.putText(img_rgb, text, (cand_box[0]+100, cand_box[1]+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color=green_rgb, thickness=1)

plt.figure(figsize=(12, 12))
plt.imshow(img_rgb)
plt.show()
