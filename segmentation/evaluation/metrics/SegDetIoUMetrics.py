from typing import Final
import torch
import os
import json
import numpy as np
import pkbar
import argparse
from shapely.geometry import Polygon
from shapely.ops import unary_union
from typing import List, Sequence, Any

def RA_to_cartesian_box(data):
    L = 4
    W = 1.8

    boxes = []
    for i in range(len(data)):

        x = np.sin(np.radians(data[i][1])) * data[i][0]
        y = np.cos(np.radians(data[i][1])) * data[i][0]

        boxes.append([x - W/2,y,x + W/2,y, x + W/2,y+L,x - W/2,y+L])

    return boxes

def perform_nms(valid_class_predictions, valid_box_predictions, nms_threshold):

    # sort the detections such that the entry with the maximum confidence score is at the top
    sorted_indices = np.argsort(valid_class_predictions)[::-1]
    sorted_box_predictions = valid_box_predictions[sorted_indices]
    sorted_class_predictions = valid_class_predictions[sorted_indices]

    for i in range(sorted_box_predictions.shape[0]):
        # get the IoUs of all boxes with the currently most certain bounding box
        try:
            IoUs = np.zeros((sorted_box_predictions.shape[0]))
            IoUs[i + 1:] = bbox_IoU(sorted_box_predictions[i, :], sorted_box_predictions[i + 1:, :])
        except ValueError:
            break
        except IndexError:
            break

        # eliminate all detections which have IoU > threshold
        overlap_mask = np.where(IoUs < nms_threshold, True, False)
        sorted_box_predictions = sorted_box_predictions[overlap_mask]
        sorted_class_predictions = sorted_class_predictions[overlap_mask]

    return sorted_class_predictions, sorted_box_predictions
def bbox_IoU(box1, boxes):

    # currently inspected box
    box1 = box1.reshape((4,2))
    rect_1 = Polygon([(box1[0, 0], box1[0, 1]), (box1[1, 0], box1[1, 1]), (box1[2, 0], box1[2, 1]),
                      (box1[3, 0], box1[3, 1])])
    area_1 = rect_1.area

    # IoU of box1 with each of the boxes in "boxes"
    IoUs = np.zeros(boxes.shape[0])
    for box_id in range(boxes.shape[0]):
        box2 = boxes[box_id]
        box2 = box2.reshape((4,2))
        rect_2 = Polygon([(box2[0, 0], box2[0, 1]), (box2[1, 0], box2[1, 1]), (box2[2, 0], box2[2, 1]),
                          (box2[3, 0], box2[3, 1])])
        area_2 = rect_2.area

        # get intersection of both bounding boxes
        inter_area = rect_1.intersection(rect_2).area

        # compute IoU of the two bounding boxes
        IoU = inter_area / (area_1 + area_2 - inter_area)

        IoUs[box_id] = IoU

    return IoUs

def process_predictions_FFT(batch_predictions, confidence_threshold=0.1, nms_threshold=0.05):

    # process targets and perform NMS for each prediction in batch
    final_batch_predictions = None  # store final bounding box predictions

    point_cloud_reg_predictions = RA_to_cartesian_box(batch_predictions)
    point_cloud_reg_predictions = np.asarray(point_cloud_reg_predictions)
    point_cloud_class_predictions = batch_predictions[:,-1]

    # get valid detections
    validity_mask = np.where(point_cloud_class_predictions > confidence_threshold, True, False)

    valid_box_predictions = point_cloud_reg_predictions[validity_mask]
    valid_class_predictions = point_cloud_class_predictions[validity_mask]


    # perform Non-Maximum Suppression
    final_class_predictions, final_box_predictions = perform_nms(valid_class_predictions, valid_box_predictions,
                                                                 nms_threshold)


    # concatenate point_cloud_id, confidence score and bounding box prediction | shape: [N_FINAL, 1+1+8]
    final_Object_predictions = np.hstack((final_class_predictions[:, np.newaxis],
                                               final_box_predictions))


    return final_Object_predictions

def GetFullMetrics(predictions,object_labels,range_min=5,range_max=100,IoU_threshold=0.5):
    raise NotImplementedError('This function is not yet implemented or verified.')
    perfs = {}
    precision = []
    recall = []
    IoU_threshold = []
    RangeError = []
    AngleError = []

    out = []

    for threshold in np.arange(0.1,0.96,0.1):

        IoU_threshold.append(threshold)

        TP = 0
        FP = 0
        FN = 0
        NbDet = 0
        NbGT = 0
        NBFrame = 0
        range_error=0
        angle_error=0
        nbObjects = 0

        for frame_id in range(len(predictions)):
            #if frame_id % 100 == 0:
            #    print(frame_id)

            pred= predictions[frame_id]
            labels = object_labels[frame_id]

            # get final bounding box predictions
            Object_predictions = []
            ground_truth_box_corners = []

            if(len(pred)>0):
                Object_predictions = process_predictions_FFT(pred,confidence_threshold=threshold)

            if(len(Object_predictions)>0):
                max_distance_predictions = (Object_predictions[:,2]+Object_predictions[:,4])/2
                ids = np.where((max_distance_predictions>=range_min) & (max_distance_predictions <= range_max) )
                Object_predictions = Object_predictions[ids]

            NbDet += len(Object_predictions)

            if(len(labels)>0):
                ids = np.where((labels[:,0]>=range_min) & (labels[:,0] <= range_max))
                labels = labels[ids]

            if(len(labels)>0):
                ground_truth_box_corners = np.asarray(RA_to_cartesian_box(labels))
                NbGT += ground_truth_box_corners.shape[0]

            # valid predictions and labels exist for the currently inspected point cloud
            if len(ground_truth_box_corners)>0 and len(Object_predictions)>0:

                used_gt = np.zeros(len(ground_truth_box_corners))
                for pid, prediction in enumerate(Object_predictions):
                    IoU = bbox_IoU(prediction[1:], ground_truth_box_corners)
                    ids = np.where(IoU>=IoU_threshold)[0]


                    if(len(ids)>0):
                        TP += 1
                        used_gt[ids]=1

                        # cummulate errors
                        range_error += np.sum(np.abs(ground_truth_box_corners[ids,-2] - prediction[-2]))
                        angle_error += np.sum(np.abs(ground_truth_box_corners[ids,-1] - prediction[-1]))
                        nbObjects+=len(ids)
                    else:
                        FP+=1
                FN += np.sum(used_gt==0)


            elif(len(ground_truth_box_corners)==0):
                FP += len(Object_predictions)
            elif(len(Object_predictions)==0):
                FN += len(ground_truth_box_corners)



        if(TP!=0):
            precision.append( TP / (TP+FP)) # When there is a detection, how much I m sure
            recall.append(TP / (TP+FN))
        else:
            precision.append( 0) # When there is a detection, how much I m sure
            recall.append(0)

        RangeError.append(range_error/nbObjects)
        AngleError.append(angle_error/nbObjects)

    perfs['precision']=precision
    perfs['recall']=recall

    F1_score = (np.mean(precision)*np.mean(recall))/((np.mean(precision) + np.mean(recall))/2)

    print('------- Detection Scores - IoU Threshold {0} ------------'.format(IoU_threshold))
    print('  mAP:',np.mean(perfs['precision']))
    print('  mAR:',np.mean(perfs['recall']))
    print('  F1 score:',F1_score)

    print('------- Regression Errors------------')
    print('  Range Error:',np.mean(RangeError),'m')
    print('  Angle Error:',np.mean(AngleError),'degree')
    #print('  Range Error: N/A')
    #print('  Angle Error: N/A')
    return [IoU_threshold,np.mean(perfs['precision']),np.mean(perfs['recall']),F1_score,np.mean(RangeError),np.mean(AngleError)]


def GetDetMetrics(predictions,object_labels,threshold=0.2,range_min=5,range_max=70,IoU_threshold=0.2):
    TP = 0
    FP = 0
    FN = 0
    NbDet=0
    NbGT=0

    range_error = 0
    angle_error = 0
    nbObjects = 0

    # get final bounding box predictions
    Object_predictions = []
    ground_truth_box_corners = []
    labels=[]

    if(len(predictions)>0):
        Object_predictions = process_predictions_FFT(predictions,confidence_threshold=threshold)

    if(len(Object_predictions)>0):
        max_distance_predictions = (Object_predictions[:,2]+Object_predictions[:,4])/2
        ids = np.where((max_distance_predictions>=range_min) & (max_distance_predictions <= range_max) )
        Object_predictions = Object_predictions[ids]

    NbDet = len(Object_predictions)

    if(len(object_labels)>0):
        ids = np.where((object_labels[:,0]>=range_min) & (object_labels[:,0] <= range_max))
        labels = object_labels[ids]
    if(len(labels)>0):
        ground_truth_box_corners = np.asarray(RA_to_cartesian_box(labels))
        NbGT = len(ground_truth_box_corners)

    # valid predictions and labels exist for the currently inspected point cloud
    if NbDet>0 and NbGT>0:

        used_gt = np.zeros(len(ground_truth_box_corners))

        for pid, prediction in enumerate(Object_predictions):
            IoU = bbox_IoU(prediction[1:], ground_truth_box_corners)
            ids = np.where(IoU>=IoU_threshold)[0]

            if(len(ids)>0):
                TP += 1
                used_gt[ids]=1

                # cummulate errors
                range_error += np.sum(np.abs(ground_truth_box_corners[ids, -2] - prediction[-2]))
                angle_error += np.sum(np.abs(ground_truth_box_corners[ids, -1] - prediction[-1]))
                nbObjects += len(ids)
            else:
                FP+=1
        FN += np.sum(used_gt==0)

    elif(NbGT==0):
        FP += NbDet
    elif(NbDet==0):
        FN += NbGT

    return TP,FP,FN, range_error, angle_error, nbObjects


# def GetSegMetrics(PredMap,label_map):
#     # Segmentation
#     pred = PredMap.reshape(-1)>=0.5
#     label = label_map.reshape(-1)
#     intersection = np.abs(pred*label).sum()
#     union = np.sum(label) + np.sum(pred) -intersection
#     IoU = intersection /union
#     return IoU

from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS
from segmentation.data.ra_encoder import ra_encoder
@METRICS.register_module()
class SegDetIoUMetrics(BaseMetric):
    def __init__(self, geometry, statistics, regression_layer=2):
        super().__init__()
        enc = ra_encoder(geometry = geometry,
                         statistics = statistics,
                         regression_layer = regression_layer)
        self.decoder = enc.decode

        # self.IoU = []
        # self.TP = 0
        # self.FP = 0
        # self.FN = 0
        # self.precision = 0
        # self.recall = 0
        # self.mIoU =0

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        pred_sem_seg = torch.stack([s['pred_sem_seg']['data'] for s in data_samples])
        pred_sem_seg = torch.sigmoid(pred_sem_seg).detach().cpu().numpy().copy()
        gt_sem_seg = torch.stack([s['gt_sem_seg']['data'] for s in data_samples]).detach().cpu().numpy().copy()
        gt_sem_seg  = gt_sem_seg.squeeze(1)
        assert (pred_sem_seg.ndim ==4 and gt_sem_seg.ndim ==3)

        pred_label_map = torch.stack([s['pred_label_map']['data'] for s in data_samples]).detach().cpu().numpy().copy()
        gt_box_label = [s['gt_box_label']['data'].detach().cpu().numpy().copy() for s in data_samples]
        assert pred_label_map.ndim ==4

        for pred_label, pred_seg, gt_label, gt_seg in zip(pred_label_map, pred_sem_seg, gt_box_label, gt_sem_seg):
            TP, FP, FN, IoU, range_error, angle_error, nbObjects = self.update(pred_seg[0], gt_seg, np.asarray(self.decoder(pred_label, 0.05)) , gt_label,
                        threshold=0.2, range_min=5, range_max=100)

            self.results.append([TP, FP, FN, IoU, range_error, angle_error, nbObjects])

    def update(self,PredMap,label_map, ObjectPred, Objectlabels, threshold=0.2, range_min=5, range_max=70):
        # Process seg
        if(len(PredMap)>0):
            pred = PredMap.reshape(-1)>=0.5
            label = label_map.reshape(-1)
            intersection = np.abs(pred*label).sum()
            union = np.sum(label) + np.sum(pred) -intersection
            IoU=(intersection /union)
        # Process det
        TP, FP, FN, range_error, angle_error, nbObjects = GetDetMetrics(ObjectPred,Objectlabels,threshold=0.2,range_min=range_min,range_max=range_max)
        return TP, FP, FN, IoU, range_error, angle_error, nbObjects

    def compute_metrics(self, results: list) -> dict:
        sum_TP = np.sum([r[0] for r in results])
        sum_FP = np.sum([r[1] for r in results])
        sum_FN = np.sum([r[2] for r in results])
        IoU = [r[3] for r in results]

        sum_range_error = np.sum([r[4] for r in results])
        sum_angle_error = np.sum([r[5] for r in results])
        sum_nbObjects = np.sum([r[6] for r in results])

        mAP, mAR, mIoU, mRE, mAE = self.GetMetrics(sum_TP, sum_FP, sum_FN, IoU, sum_range_error, sum_angle_error, sum_nbObjects)

        val_metrics = {'mean_AP': mAP,
                    'mean_AR': mAR,
                    'mean_IoU': mIoU,
                    'mean_range_error': mRE,
                    'mean_angle_error': mAE
                       }
        return val_metrics

    def GetMetrics(self, TP, FP, FN, IoU:list, sum_range_error, sum_angle_error, sum_nbObjects) -> dict:
        if( TP + FP != 0):
            precision = TP / (TP + FP)
        if( TP + FN != 0):
            recall = TP / (TP + FN)
        if(len(IoU)>0):
            mIoU = np.asarray(IoU).mean()

        mRE = sum_range_error/sum_nbObjects
        mAE = sum_angle_error / sum_nbObjects

        return precision, recall, mIoU, mRE, mAE

    # def reset(self,):
    #     self.IoU = []
    #     self.TP = 0
    #     self.FP = 0
    #     self.FN = 0
    #     self.precision = 0
    #     self.mIoU =0
