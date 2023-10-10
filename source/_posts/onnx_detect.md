---
title: decorator
date: 2023-08-10 14:51:35
type: "tags"
tags:
    -html
    -onnx
    -detect
keywords: '深度学习目标检测'
cover: https://s2.loli.net/2023/08/10/qckoIznlQp4Ot69.jpg
---
训练好的深度学习模型通常需要转换为onnx模型，官方有onnx的测试代码，这里重新进行构建，并手动生成anchor文件，进行onnx模型测试，对于输入图像前处理也分为两部分，一个包含letterbox；一个不包含直接resize，显然后一个精度会有多降低
# 后处理
{% codeblock [后处理]  %}
import onnxruntime
import cv2
import numpy as np
import torch
#---------------------NMS--------------------------------------------------------------------
def py_cpu_nms(dets0, conf_thresh, iou_thresh):
    """Pure Python NMS baseline."""
    nc = dets0.shape[1] - 5
    dets = dets0[dets0[:, 4] > conf_thresh]
    dets = xywh2xyxy(dets)
    
    keep_all = []
    for cls in range(nc):
        dets_single = dets[np.argmax(dets[:,5:],axis=1)==cls]
        #print('dets_single %d'%cls,dets_single)
        x1 = dets_single[:, 0]
        y1 = dets_single[:, 1]
        x2 = dets_single[:, 2]
        y2 = dets_single[:, 3]
        scores = dets_single[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1) 
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)  
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_thresh)[0]
            order = order[inds + 1]
        keep_rect = dets_single[keep]
        #print('keep',keep)
        keep_all.extend(keep_rect)
    return keep_all
 
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    y[:, 4:] = x[:,4:]
    return y

#---------------------img_preprocess-----------------------------------------------------------------
def img_preprocess(frame,imgsz):
    # im = letterbox(frame, imgsz)[0]
    im = cv2.resize(frame,(640,384))
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    im = np.asarray(im, dtype=np.float32)
    im = np.expand_dims(im, 0)
    im /= 255.0
    return im

#---------------------------letterbox------------------------------------------------------------------
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

#-----------------------------decode----------------------------------------
def np_sigmoid(x):
    return 1.0/(1.0+1.0/np.exp(x))
    
def decode_output(pred_raw_data,anchor_txt):
    pred_raw_data = np_sigmoid(pred_raw_data)
    print(max(pred_raw_data[:, 4]))
    pred_raw_data[:, 0] = (pred_raw_data[:, 0] * 2. - 0.5 + anchor_txt[:, 0]) * anchor_txt[:, 4] #x
    pred_raw_data[:, 1] = (pred_raw_data[:, 1] * 2. - 0.5 + anchor_txt[:, 1]) * anchor_txt[:, 4] #y
    pred_raw_data[:, 2] = (pred_raw_data[:, 2] * 2) ** 2 * anchor_txt[:, 2]  # w
    pred_raw_data[:, 3] = (pred_raw_data[:, 3] * 2) ** 2 * anchor_txt[:, 3]  # h
    
    return pred_raw_data
    
#-------------------------scale_ratio------------------------------------------------------
def helmet_scale_ratio(each,frame,imgsz=640):
    ratio = (frame.shape[0] /384 , frame.shape[1] / imgsz)
    each[[0, 2]] *= ratio[1]
    each[[1, 3]] *= ratio[0]
    return each
     
#---------------------helmet_detect----------------------------------------------------------------------------
def helmet_detect(face_model,frame):
    anchors = np.fromfile('C:/Users/suso/Desktop/yolov5_fire/yolov5_fire_priorbox_384-640.txt',sep=' ')
    anchors = anchors.reshape(-1,5)
    imgsz = 640
    img = img_preprocess(frame,imgsz) 

    session = onnxruntime.InferenceSession(face_model)
    in_name = [input.name for input in session.get_inputs()][0]
    out_name = [output.name for output in session.get_outputs()]              
    pred = session.run(out_name,{in_name: img})  
    
    
    x1 = np.array(pred[0]).reshape(-1, 6)
    x2 = np.array(pred[1]).reshape(-1, 6)
    x3 = np.array(pred[2]).reshape(-1, 6)
    print(x3.shape,max(x3[:,4]))

    try:
        # Save x1 to file
        np.savetxt("C:/Users\suso\Desktop/yolov5_fire/x1.txt", x1, delimiter=' ', fmt='%f')
        # Save x2 to file
        np.savetxt("C:/Users\suso\Desktop/yolov5_fire/x2.txt", x2, delimiter=' ', fmt='%f')
        # Save x3 to file
        np.savetxt("C:/Users\suso\Desktop/yolov5_fire/x3.txt", x3, delimiter=' ', fmt='%f')
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    out_data_raw = np.vstack((x1,x2,x3))
    np.savetxt("C:/Users\suso\Desktop/yolov5_fire/x4.txt", out_data_raw, delimiter=' ', fmt='%f')
    output_from_txt = decode_output(out_data_raw,anchors)

    print(len(pred),pred[0].shape)
    print("ffffffffff",max(np.array(pred[0]).reshape(-1, 6)[:,4]))
    # pred = py_cpu_nms(np.array(pred[0]).reshape(-1, 6), 0.2, 0.45)
    pred = py_cpu_nms(output_from_txt, 0.2, 0.45)
    
    return pred,img

#---------------draw--------------------------------------------
def drawHelmetBox(frame,bbox):
    print(bbox)
    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 3)
    label = f'{float(bbox[4]*bbox[5]):.2f}'
    cv2.putText(frame, label, (int(bbox[0]), int(bbox[1])+20), 0, 1, [0, 255, 0], thickness=2, lineType=cv2.LINE_AA)
    cv2.imshow('frame',frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
{% endcodeblock %}

# 测试
{% codeblock [测试]  %}
import cv2
import os
from HelmetDetection import helmet_detect,helmet_scale_ratio,drawHelmetBox,scale_boxes

# image_path = 'C:/Users/suso/Desktop/yolov5_fire/frame_01_jpg.rf.7a9e0fe6e03efe5b3df01c7322aff0dc.jpg'
face_model = 'C:/Users/suso/Desktop/yolov5_fire/best.onnx'
for imgname in os.listdir('C:/Users/suso/Desktop/yolov5_fire/test_jpg'):
    print(imgname)
    image_path = 'C:/Users/suso/Desktop/yolov5_fire/test_jpg/'+imgname
    frame = cv2.imread(image_path)
    helmet_pred,img = helmet_detect(face_model,frame)
    print(helmet_pred)
    for bbox in helmet_pred:
        bbox = helmet_scale_ratio(bbox,frame)#no letterbox
        # im_shape = img.shape[2:]
        # frame_shape = frame.shape
        # bbox = scale_boxes(im_shape, bbox, frame_shape)#add letterbox
        drawHelmetBox(frame,bbox)


{% endcodeblock %}

