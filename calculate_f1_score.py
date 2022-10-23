from pathlib import Path
from sklearn.metrics import f1_score
import cv2
import json


label_path = Path(
    'runs/detect/exp/labels')
ground_truth_path = Path(
    '../test_set/Annotations')


def parse_labels(raw_labels):
    result = []
    for l in raw_labels.split('\n'):
        if(len(l) > 0):
            splitted = l.split(' ')
            result.append({
                'class': splitted[0],
                'score': float(splitted[1]),
                'lt_x': int(splitted[2]),
                'lt_y': int(splitted[3]),
                'rb_x': int(splitted[4]),
                'rb_y': int(splitted[5]),
            })
    return result


def parse_ground_truth(root):
    result = []
    for objtemp in root.findall('object'):
        xmin = objtemp.find('bndbox').find('xmin').text
        result.append({
            'class': objtemp.find('name').text,
            'lt_x': int(objtemp.find('bndbox').find('xmin').text),
            'lt_y': int(objtemp.find('bndbox').find('ymin').text),
            'rb_x': int(objtemp.find('bndbox').find('xmax').text),
            'rb_y': int(objtemp.find('bndbox').find('ymax').text),
        })
    return result


def calc_overlap_area(rec1, rec2):
    lt_x = max(rec1['lt_x'], rec2['lt_x'])
    lt_y = max(rec1['lt_y'], rec2['lt_y'])
    rb_x = min(rec1['rb_x'], rec2['rb_x'])
    rb_y = min(rec1['rb_y'], rec2['rb_y'])
    width_x = max(0, rb_x - lt_x)
    width_y = max(0, rb_y - lt_y)
    return width_x * width_y


def calc_iou(rec1, rec2):
    area1 = (rec1['rb_x'] - rec1['lt_x']) * (rec1['rb_y'] - rec1['lt_y'])
    area2 = (rec2['rb_x'] - rec2['lt_x']) * (rec2['rb_y'] - rec2['lt_y'])
    overlap = calc_overlap_area(rec1, rec2)
    return overlap / (area1 + area2 - overlap)


def calc_f1_score(labels, ground_truth):
    label_name = ['bg', 'helmet', 'head']
    lable_true = []
    lable_pred = []
    for g in ground_truth:
        iou_max = 0
        iou_max_index = -1
        for i, l in enumerate(labels):
            iou = calc_iou(g, l)
            if(iou > iou_max):
                iou_max = iou
                iou_max_index = i
        if(iou_max > 0.5):
            lable_true.append(label_name.index(g['class']))
            lable_pred.append(label_name.index(labels[iou_max_index]['class']))
            labels.pop(iou_max_index)
        else:
            lable_true.append(label_name.index(g['class']))
            lable_pred.append(0)
    for l in labels:
        lable_true.append(0)
        lable_pred.append(label_name.index(l['class']))
    return f1_score(lable_true, lable_pred, average='micro'), lable_true, lable_pred


def main():
    lable_true_list = []
    lable_pred_list = []
    f1_list = []
    for p in label_path.iterdir():
        print(p)        
        import xml.etree.ElementTree as ET
        tree = ET.parse(ground_truth_path / (p.stem + '.xml'))
        root = tree.getroot()
        ground_truth = parse_ground_truth(root)
        with open(p, 'r') as fp:
            labels = parse_labels(fp.read())
        f1, lable_true, lable_pred = calc_f1_score(labels, ground_truth)
        f1_list.append(f1)
        lable_true_list += lable_true
        lable_pred_list += lable_pred
    print(sum(f1_list) / len(f1_list), f1_score(lable_true_list, lable_pred_list, average='micro'))


if __name__ == "__main__":
    main()
