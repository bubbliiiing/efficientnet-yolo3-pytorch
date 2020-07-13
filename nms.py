import numpy as np

def non_max_suppression(boxes, num_classes, conf_thres=0.5, nms_thres=0.4):
    bs = np.shape(boxes)[0]
    # 将框转换成左上角右下角的形式
    shape_boxes = np.zeros_like(boxes[:,:,:4])
    shape_boxes[:,:,0] = boxes[:,:,0] - boxes[:,:,2]/2
    shape_boxes[:,:,1] = boxes[:,:,1] - boxes[:,:,3]/2
    shape_boxes[:,:,2] = boxes[:,:,0] + boxes[:,:,2]/2
    shape_boxes[:,:,3] = boxes[:,:,1] + boxes[:,:,3]/2

    boxes[:,:,:4] = shape_boxes
    output = []
    # 1、对所有图片进行循环。
    for i in range(bs):
        prediction = boxes[i]
        # 2、找出该图片中得分大于门限函数的框。在进行重合框筛选前就进行得分的筛选可以大幅度减少框的数量。
        mask = prediction[:,4] >= conf_thres
        prediction = prediction[mask]
        if not np.shape(prediction)[0]:
            continue

        # 3、判断第2步中获得的框的种类与得分。
        # 取出预测结果中框的位置与之进行堆叠。
        # 此时最后一维度里面的内容由5+num_classes变成了4+1+2，
        # 四个参数代表框的位置，一个参数代表预测框是否包含物体，两个参数分别代表种类的置信度与种类。
        class_conf = np.expand_dims(np.max(prediction[:, 5:5 + num_classes], 1),-1)
        class_pred = np.expand_dims(np.argmax(prediction[:, 5:5 + num_classes], 1),-1)
        detections = np.concatenate((prediction[:, :5], class_conf, class_pred), 1)
        unique_class = np.unique(detections[:,-1])
        
        if len(unique_class) == 0:
            continue
        
        best_box = []
        # 4、对种类进行循环，
        # 非极大抑制的作用是筛选出一定区域内属于同一种类得分最大的框，
        # 对种类进行循环可以帮助我们对每一个类分别进行非极大抑制。
        for c in unique_class:
            cls_mask = detections[:,-1] == c

            detection = detections[cls_mask]
            scores = detection[:,4]
            # 5、根据得分对该种类进行从大到小排序。
            arg_sort = np.argsort(scores)[::-1]
            detection = detection[arg_sort]
            print(detection)
            while np.shape(detection)[0]>0:
                # 6、每次取出得分最大的框，计算其与其它所有预测框的重合程度，重合程度过大的则剔除。
                best_box.append(detection[0])
                if len(detection) == 1:
                    break
                ious = iou(best_box[-1],detection[1:])
                detection = detection[1:][ious<nms_thres]
        output.append(best_box)
    return np.array(output)

def iou(b1,b2):
    b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)
    
    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
                 np.maximum(inter_rect_y2 - inter_rect_y1, 0)
    
    area_b1 = (b1_x2-b1_x1)*(b1_y2-b1_y1)
    area_b2 = (b2_x2-b2_x1)*(b2_y2-b2_y1)
    
    iou = inter_area/np.maximum((area_b1+area_b2-inter_area),1e-6)
    return iou
