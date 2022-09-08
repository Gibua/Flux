from ctypes import pointer
import os.path as osp
from re import A
import scipy.io as sio
import numpy as np
from math import dist

import time

from utils.face_detection import unwrap_bbox
from glue import RetinaFace, PIPNet, ULFace
from utils.face_detection import bbox_from_landmark, bbox_intersect, bbox_area

import cv2

BASE_DATA_PATH = osp.expanduser("~/datasets/300W/")

def compute_nme(lms_pred, lms_gt, norm):
    lms_pred = lms_pred.reshape((-1, 2))
    lms_gt = lms_gt.reshape((-1, 2))
    nme = np.mean(np.linalg.norm(lms_pred - lms_gt, axis=1)) / norm 
    return nme

def non_max_suppression_fast(boxes, probs=None, overlapThresh=0.3):

    # if there are no boxes, return an empty list
    if boxes.size == 0:
        return np.empty(0)
    if boxes.shape[0] == 1:
        return boxes.astype(int)

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1, y1, x2, y2 = boxes.T

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked
    return boxes[pick].astype("int")

def get_marks_from_pts(pts_file):
    points = []
    with open(pts_file, 'r') as file:
        while file.readline()[:-1] != '{':
            continue
        while True:
            line = file.readline()[:-1]
            if line == "}":
                break
            space_idx = line.find(" ")
            points.append([float(line[:space_idx]),float(line[space_idx:])])
    return np.array(points)


detector = RetinaFace.Detector()
fallback = ULFace.Detector()
lm_predictor = PIPNet.Predictor('300W_COFW_WFLW')

if lm_predictor.landmark_count == 68:
    dataset = 'iBUG'
elif lm_predictor.landmark_count == 98:
    dataset = 'WFLW'

#cv2.namedWindow("Display", cv2.WINDOW_NORMAL)
all_pts = {}
#for i in range(1, 301):
i=1
nme_sum=0
while i<=300:
    file_name = 'outdoor_{:03d}'.format(i)
    image_name = file_name+".png"
    
    image_path = osp.join(BASE_DATA_PATH, "02_Outdoor/" + file_name + ".png")
    pts_path = osp.join(BASE_DATA_PATH, "02_Outdoor/" + file_name + ".pts")
    
    all_pts[file_name+".png"] = get_marks_from_pts(pts_path)

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    gt_points = all_pts[image_name]
    gt_points_bbox = bbox_from_landmark(gt_points)

    ret_1, bboxes_1 = detector.detect_bbox(img)
    ret_2, bboxes_2 = fallback.detect_bbox(img)
    
    if ret_1 and ret_2:
        bboxes = np.concatenate([bboxes_1[:,:4], bboxes_2])
    elif ret_1:
        bboxes = bboxes_1[:,:4]
    else:
        bboxes = bboxes_2
    
    print(ret_1, ret_2)
    bboxes = non_max_suppression_fast(bboxes, overlapThresh=0.4)   

    ret = ret_1 or ret_2
    
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = unwrap_bbox(bbox.astype(np.int32))
        cv2.rectangle(img, (xmin, ymin),
                           (xmax, ymax), (0, 0, 255), 2)

    xmin, ymin, xmax, ymax = unwrap_bbox(gt_points_bbox.astype(np.int32))
    cv2.rectangle(img, (xmin, ymin),
                       (xmax, ymax), (150, 250, 0), 2)

    #print(ret,file_name)

    bbox_idx = 0
    last_ratio = 0
    
    for idx in range(len(bboxes)):
        gt_bbox_area = bbox_area(gt_points_bbox)
        current_bbox_area = bbox_area(bboxes[idx])

        overlap = bbox_intersect(gt_points_bbox, bboxes[idx])
        overlapping_area = bbox_area(overlap)

        overlap_proportion = overlapping_area/max(gt_bbox_area, current_bbox_area)
        if (overlap_proportion > last_ratio):
            bbox_idx = idx
            last_ratio = overlap_proportion
    
    
    bbox = (bboxes[bbox_idx]).astype(int)
    
    out = lm_predictor.predict(img, bbox)
    
    preds = lm_predictor.post_process(out, bbox)
    
    norm_indices = [36, 45]
    if dataset == "iBUG" or dataset == "WFLW":
        norm = np.linalg.norm(gt_points.reshape(-1, 2)[norm_indices[0]] - gt_points.reshape(-1, 2)[norm_indices[1]])
    elif dataset == "AFLW":
        norm = 1

    nme = compute_nme(preds, gt_points, norm)
    nme_sum += nme
    print(image_name, "nme: ", nme)

    for (x, y) in gt_points:
        cv2.circle(img, (np.int32(x), np.int32(y)), 1, (125, 255, 0))

    for (x, y) in preds:
        cv2.circle(img, (np.int32(x), np.int32(y)), 4, (0, 0, 255))

    #cv2.imshow('Display', img)

    #k = cv2.waitKey(0) & 0xFF
    #print(k)
    #if k == 27:
    #    break
    #if k == 81:
    #    i -= 1
    #    if i<213:
    #        i=213
    #elif k == 83:
    #    i += 1
    b = "Loading" + ("." * (i%4))
    print (b, end="\r")
    i += 1

print(nme_sum/300.0)

cv2.destroyAllWindows()