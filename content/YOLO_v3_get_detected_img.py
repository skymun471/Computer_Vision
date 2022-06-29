# OpenCV Darknet Yolo를 이용하여 이미지 Object Detection

# 입력 이미지로 사용될 이미지 다운로드 wget을 활용하여 github주소에 있는 것을 다운로드한다.
# !wget -o ./data/beatles01.jpg <주소 입력 > -O 옵션을 주면 자신이 원하는 위치에 원하는 이름으로 파일을 저장할 수 있다.

# Darknet Yolo 사이트에서 coco로 학습된 inference model을 다운로드한후 OpenCv에서 infernece model 생성한다.
# yolov3 weigh와 cfg 다운로드
# !wget -O ./yolov3.cfg https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true
# !wget -O ./yolov3.weights https://pjreddie.com/media/files/yolov3.weights

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time



def OpenCV_v3_get_detected_img(cv_net, img_array, conf_threshold, nms_threshold, is_print = True):
    cv_net_yolo = cv_net
    #  coco class_id와 class명 맵핑
    #  COCO는 총 80개의 클래스로 이루어져있다.
    labels_to_names_seq = {0:'person',1:'bicycle',2:'car',3:'motorbike',4:'aeroplane',5:'bus',6:'train',7:'truck',8:'boat',9:'traffic light',10:'fire hydrant',
                            11:'stop sign',12:'parking meter',13:'bench',14:'bird',15:'cat',16:'dog',17:'horse',18:'sheep',19:'cow',20:'elephant',
                            21:'bear',22:'zebra',23:'giraffe',24:'backpack',25:'umbrella',26:'handbag',27:'tie',28:'suitcase',29:'frisbee',30:'skis',
                            31:'snowboard',32:'sports ball',33:'kite',34:'baseball bat',35:'baseball glove',36:'skateboard',37:'surfboard',38:'tennis racket',39:'bottle',40:'wine glass',
                            41:'cup',42:'fork',43:'knife',44:'spoon',45:'bowl',46:'banana',47:'apple',48:'sandwich',49:'orange',50:'broccoli',
                            51:'carrot',52:'hot dog',53:'pizza',54:'donut',55:'cake',56:'chair',57:'sofa',58:'pottedplant',59:'bed',60:'diningtable',
                            61:'toilet',62:'tvmonitor',63:'laptop',64:'mouse',65:'remote',66:'keyboard',67:'cell phone',68:'microwave',69:'oven',70:'toaster',
                            71:'sink',72:'refrigerator',73:'book',74:'clock',75:'vase',76:'scissors',77:'teddy bear',78:'hair drier',79:'toothbrush' }

    ## 3개의 scale Output layer에서 결과 데이터 도츨
    layer_names = cv_net_yolo.getLayerNames()
    # print('### yolo v3 layer name:', layer_names)
    # print('final output layer id:', cv_net_yolo.getUnconnectedOutLayers())
    # print('final output layer name:', [layer_names[i - 1] for i in cv_net_yolo.getUnconnectedOutLayers()])

    output_layer_names = [layer_names[i-1] for i in cv_net_yolo.getUnconnectedOutLayers()]

    img = cv2.imread(img_array)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 원본 이미지를 Yolo_v3 416X416 model임, 원본 이미지 크기를 416 X 416으로, bgr을 RGB로 설정히여 입력으로 넣어준다. 
    cv_net_yolo.setInput(cv2.dnn.blobFromImage(img, scalefactor = 1/255.0, size=(416,416),swapRB = True, crop =False))
    start = time.time()
    # odject_detection 결과를 CVoUT으로 빈환
    # cv_outs으로 3개의 LAYER에 대힌 결과가 나온다. 
    cv_outs = cv_net_yolo.forward(output_layer_names)
    # print('cv_outs_type:', type(cv_outs), 'cv_outs의 내부 원소개수:', len(cv_outs))
    # print(cv_outs[0].shape, cv_outs[1].shape, cv_outs[2].shape)
    # print(cv_outs)

    # object detection 정보를 모두 수집
    # center와 width, height 정보를 모두 좌상단, 우하단 좌표로 변경 
    # =============================================================================================================================
    # 원본 이미지를 network에 입력하기 위해서는 416 x 416 size로 resize해야한다.
    # 이후 결과가 출력되면 resize 기반으로 되어있기 때문에 복원하기 위해서 원본 이미지 shape가 필요
    rows = img.shape[0]
    cols = img.shape[1]

    conf_threshold = conf_threshold
    nms_threshold = nms_threshold

    # bounding_bo와 테두리  caption 글자색 지정
    green_color = (0,255,0)
    red_color = (0,0,255)

    class_ids = []
    confidences = []
    boxes=[]

    # 3개의 개별 out layer 별로 detection 된 object 별로 정보 추출 및 시각화 
    for ix , output in enumerate(cv_outs):
        # print('output_ shape', output.shape)
        for jx , detection in enumerate(output):
            class_scores = detection[5:]
            # class_score가 가장 높은 인덱스 번호를 argmax가 추출하여 이를 class_id로 지정한다.
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            
            if confidence > conf_threshold:
                # print('ix',ix,'jx',jx, 'class_id', class_id, 'confidence',confidence)
                center_x = int(detection[0] * cols)
                center_y = int(detection[1] * rows)
                width = int(detection[2] * cols)
                height = int(detection[3]* rows)
                left = int(center_x - width/2)
                top = int(center_y - height/2)
                
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                
    # nms를 이용하여 각 Output layer에서 detected된 Bbox를 제외
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    # print(idxs.flatten())
    # print(idxs.flatten())


    draw_img = img.copy()
    if len(idxs) > 0:
        for i in idxs.flatten():
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            
            caption = "{}: {:.4f}".format(labels_to_names_seq[class_ids[i]], confidences[i])
            cv2.rectangle(draw_img,(int(left),int(top)), (int(left+width), int(top+height)), color = green_color, thickness=2)
            cv2.putText(draw_img, caption, (int(left), int(top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red_color, 1)
            print(caption)

    if is_print == True:
        print("Detection 수행시간:", round(time.time() - start, 2),"초")
    return draw_img



# ============================================================================================================================
#test




weight_path = '/home/haneul/Desktop/DLCV/Computer_Vision/content/pretrained/yolov3.weights'
cfg_path = '/home/haneul/Desktop/DLCV/Computer_Vision/content/pretrained/yolov3.cfg'
img_array = '/home/haneul/Desktop/DLCV/Computer_Vision/content/data/baseball01.jpg'
cv_net = cv2.dnn.readNetFromDarknet(cfg_path, weight_path)
conf_threshold = 0.5
nms_threshold = 0.4

draw_img = OpenCV_v3_get_detected_img(cv_net, img_array, conf_threshold, nms_threshold, is_print = True)

img_rgb = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12,12))
plt.imshow(img_rgb)
plt.show() 