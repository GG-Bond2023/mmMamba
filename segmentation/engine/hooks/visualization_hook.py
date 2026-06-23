import os.path as osp
import warnings
from typing import Optional, Sequence

import mmcv, os
import mmengine.fileio as fileio
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.visualization import Visualizer

from mmseg.registry import HOOKS
from mmseg.structures import SegDataSample

from mmseg.engine.hooks.visualization_hook import SegVisualizationHook

from mmengine.model import MMDistributedDataParallel

import torch
import numpy as np
import cv2
from shapely.geometry import Polygon
from shapely.ops import unary_union
import polarTransform

# Camera parameters
camera_matrix = np.array([[1.84541929e+03, 0.00000000e+00, 8.55802458e+02],
                 [0.00000000e+00 , 1.78869210e+03 , 6.07342667e+02],[0.,0.,1]])
dist_coeffs = np.array([2.51771602e-01,-1.32561698e+01,4.33607564e-03,-6.94637533e-03,5.95513933e+01])
rvecs = np.array([1.61803058, 0.03365624,-0.04003127])
tvecs = np.array([0.09138029,1.38369885,1.43674736])
ImageWidth = 1920
ImageHeight = 1080


@HOOKS.register_module()
class RadialVisualizationHook(SegVisualizationHook):
    """Segmentation Visualization Hook. Used to visualize validation and
    testing process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """
    def __init__(self,**kwargs):
        self.draw = kwargs.pop('draw', False)
        self.interval = kwargs.pop('interval', 50)
        self.show = kwargs.pop('show', False)
        self.wait_time = kwargs.pop('wait_time', 0.)
        self.backend_args = kwargs.pop('backend_args', None)

        self.threshold = kwargs.pop('threshold',0.5)
        self.mapping_color = kwargs.pop('mapping_color',[46, 206, 254])
        self.ksize = kwargs.pop('ksize',(3,3))
        self.dilated_iter = kwargs.pop('dilated_iter',5)

        super().__init__(draw= self.draw, interval= self.interval,show=self.show, wait_time=self.wait_time, backend_args=self.backend_args)

    def _after_iter(self,
                    runner: Runner,
                    batch_idx: int,
                    data_batch: dict,
                    outputs: Sequence[SegDataSample],
                    mode: str = 'val') -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]): Outputs from models.
            mode (str): mode (str): Current mode of runner. Defaults to 'val'.
        """
        if self.draw is False or mode == 'train':
            return

        if self.every_n_inner_iters(batch_idx, self.interval):
            for idx in range(len(outputs)):
                output = outputs[idx]
                window_name = f'{mode}_{osp.basename(output.camera_path)}'
                if runner.cfg['data_mode'] == 'ADC':

                    # Get the original model (compatible with single/multi GPU)
                    model_t = runner.model
                    if isinstance(model_t, MMDistributedDataParallel):
                        model_t = model_t.module  # Get the original model in distributed environment

                    intermediate = model_t.backbone.DFT(data_batch['inputs'][idx].unsqueeze(0).to(output.gt_sem_seg.data.device)).detach().cpu().numpy()[0]
                else:
                    intermediate = None
                # hmi = DisplayHMI(
                #         image = output.gt_image.data,
                #         input = data_batch['inputs'][idx].permute(1,2,0),
                #         model_outputs = output,
                #         config =runner.cfg,
                #         decode = runner.val_dataloader.dataset.decoder,
                #         intermediate = intermediate,
                # )
                hmi = DisplayHMI_with_SemMap(
                        image = output.gt_image.data.detach().cpu(),
                        input = data_batch['inputs'][idx].permute(1,2,0).detach().cpu(),
                        model_outputs = output.detach().cpu(),
                        config =runner.cfg,
                        decode = runner.val_dataloader.dataset.decoder,
                        segmap = output.gt_sem_seg.data.detach().cpu(),
                        intermediate = intermediate,

                        threshold=self.threshold,
                        mapping_color=self.mapping_color,
                        ksize=self.ksize,
                        dilated_iter=self.dilated_iter
                )
                if mode=='test':
                    save_folder = os.path.join(runner.log_dir,mode)
                else:
                    save_folder = os.path.join(runner.log_dir, mode, 'iter_'+str(runner.iter))
                os.makedirs(save_folder, exist_ok=True)
                save_path = os.path.join(save_folder, window_name)
                success = cv2.imwrite(save_path, hmi)
                if success:
                    print(f"{mode} image successfully saved to: {save_path}")
                else:
                    print("Save failed! Please check if the path is correct or the image is valid.")



def worldToImage(x,y,z):
    world_points = np.array([[x,y,z]],dtype = 'float32')
    rotation_matrix = cv2.Rodrigues(rvecs)[0]

    imgpts, _ = cv2.projectPoints(world_points, rotation_matrix, tvecs, camera_matrix, dist_coeffs)

    u = int(min(max(0,imgpts[0][0][0]),ImageWidth-1))
    v = int(min(max(0,imgpts[0][0][1]),ImageHeight-1))

    return u,v

def RA_to_cartesian_box(data):
    L = 4
    W = 1.8

    boxes = []
    for i in range(len(data)):

        x = np.sin(np.radians(data[i][1])) * data[i][0]
        y = np.cos(np.radians(data[i][1])) * data[i][0]

        boxes.append([x - W/2,y,x + W/2,y, x + W/2,y+L,x - W/2,y+L,data[i][0],data[i][1]])

    return boxes

def perform_nms(valid_class_predictions, valid_box_predictions, nms_threshold):

    # sort the detections such that the entry with the maximum confidence score is at the top
    sorted_indices = np.argsort(valid_class_predictions)[::-1]
    sorted_box_predictions = valid_box_predictions[sorted_indices]
    sorted_class_predictions = valid_class_predictions[sorted_indices]

    for i in range(sorted_box_predictions.shape[0]):
        # get the IOUs of all boxes with the currently most certain bounding box
        try:
            ious = np.zeros((sorted_box_predictions.shape[0]))
            ious[i + 1:] = bbox_iou(sorted_box_predictions[i, :8], sorted_box_predictions[i + 1:, :8])
        except ValueError:
            break
        except IndexError:
            break

        # eliminate all detections which have IoU > threshold
        overlap_mask = np.where(ious < nms_threshold, True, False)
        sorted_box_predictions = sorted_box_predictions[overlap_mask]
        sorted_class_predictions = sorted_class_predictions[overlap_mask]

    return sorted_class_predictions, sorted_box_predictions

def bbox_iou(box1, boxes):

    # currently inspected box
    box1 = box1.reshape((4,2))
    rect_1 = Polygon([(box1[0, 0], box1[0, 1]), (box1[1, 0], box1[1, 1]), (box1[2, 0], box1[2, 1]),
                      (box1[3, 0], box1[3, 1])])
    area_1 = rect_1.area

    # IoU of box1 with each of the boxes in "boxes"
    ious = np.zeros(boxes.shape[0])
    for box_id in range(boxes.shape[0]):
        box2 = boxes[box_id]
        box2 = box2.reshape((4,2))
        rect_2 = Polygon([(box2[0, 0], box2[0, 1]), (box2[1, 0], box2[1, 1]), (box2[2, 0], box2[2, 1]),
                          (box2[3, 0], box2[3, 1])])
        area_2 = rect_2.area

        # get intersection of both bounding boxes
        inter_area = rect_1.intersection(rect_2).area

        # compute IoU of the two bounding boxes
        iou = inter_area / (area_1 + area_2 - inter_area)

        ious[box_id] = iou

    return ious


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
    final_class_predictions, final_box_predictions = perform_nms(valid_class_predictions, valid_box_predictions,nms_threshold)

    # concatenate point_cloud_id, confidence score and bounding box prediction | shape: [N_FINAL, 1+1+8]
    final_point_cloud_predictions = np.hstack((final_class_predictions[:, np.newaxis],
                                               final_box_predictions))
    return final_point_cloud_predictions

def DisplayHMI(image, input, model_outputs,decode,config,intermediate=None):
    image = image.detach().cpu().numpy().copy()
    input = input.detach().cpu().numpy().copy()
    # Model outputs
    pred_obj = model_outputs.pred_label_map.data.detach().cpu().numpy().copy()
    out_seg = torch.sigmoid(model_outputs.pred_sem_seg.data).detach().cpu().numpy().copy()[0]

    # Decode the output detection map
    pred_obj = decode(pred_obj,0.05)
    pred_obj = np.asarray(pred_obj)

    # process prediction: polar to cartesian, NMS...
    if(len(pred_obj)>100):
        print("Are you using an untrained pth? This will cause this line of code to take too long!!! (segmentation/engine/hooks/visualization_hook.py)")
        # print("pred_obj count > 20, only using first 20 (segmentation/engine/hooks/visualization_hook.py)")
        # pred_obj = pred_obj[:20]
    if(len(pred_obj)>0):
        pred_obj = process_predictions_FFT(pred_obj,confidence_threshold=0.2)

    ## FFT
    if config['data_mode'] != 'ADC':
        FFT = np.abs(input[...,:16]+input[...,16:]*1j).mean(axis=2)
    else:
        FFT = np.abs(intermediate[:16,:,:]+intermediate[16:,:,:]*1j).mean(axis=0)

    PowerSpectrum = np.log10(FFT)
    # rescale
    PowerSpectrum = (PowerSpectrum -PowerSpectrum.min())/(PowerSpectrum.max()-PowerSpectrum.min())*255
    PowerSpectrum = cv2.cvtColor(PowerSpectrum.astype('uint8'),cv2.COLOR_GRAY2BGR)

    if image.flags.writeable is False:
        image = image.copy()

    ## Image
    for box in pred_obj:
        box = box[1:]
        u1,v1 = worldToImage(-box[2],box[1],0)
        u2,v2 = worldToImage(-box[0],box[1],1.6)

        u1 = int(u1/2)
        v1 = int(v1/2)
        u2 = int(u2/2)
        v2 = int(v2/2)

        image = cv2.rectangle(image, (u1,v1), (u2,v2), (0, 0, 255), 3)

    RA_cartesian,_=polarTransform.convertToCartesianImage(np.moveaxis(out_seg,0,1),useMultiThreading=True,
        initialAngle=0, finalAngle=np.pi,order=0,hasColor=False)

    # Make a crop on the angle axis
    RA_cartesian = RA_cartesian[:,256-100:256+100]

    RA_cartesian = np.asarray((RA_cartesian*255).astype('uint8'))
    RA_cartesian = cv2.cvtColor(RA_cartesian, cv2.COLOR_GRAY2BGR)
    RA_cartesian = cv2.resize(RA_cartesian,dsize=(400,512))
    RA_cartesian=cv2.flip(RA_cartesian,flipCode=-1)

    return np.hstack((PowerSpectrum,image[:512],RA_cartesian))

def DisplayHMI_with_SemMap(image, input, model_outputs,decode, config, segmap, intermediate=None, threshold=0.5, mapping_color=[46, 206, 254], ksize=(3,3), dilated_iter=5):

    image = image.detach().cpu().numpy().copy()
    input = input.detach().cpu().numpy().copy()
    # Model outputs
    pred_obj = model_outputs.pred_label_map.data.detach().cpu().numpy().copy()
    out_seg = torch.sigmoid(model_outputs.pred_sem_seg.data).detach().cpu().numpy().copy()[0]

    # Decode the output detection map
    pred_obj = decode(pred_obj,0.05)
    pred_obj = np.asarray(pred_obj)

    # process prediction: polar to cartesian, NMS...
    if(len(pred_obj)>100):
        # print("Are you using an untrained pth? This will cause this line of code to take too long!!! (segmentation/engine/hooks/visualization_hook.py)")
        print("Number of pred_obj exceeds 20, using only the first 20 (segmentation/engine/hooks/visualization_hook.py)")
        pred_obj = pred_obj[:20,:]
    if(len(pred_obj)>0):
        pred_obj = process_predictions_FFT(pred_obj,confidence_threshold=0.2)

    ## FFT
    if config['data_mode'] != 'ADC':
        FFT = np.abs(input[...,:16]+input[...,16:]*1j).mean(axis=2)
    else:
        FFT = np.abs(intermediate[:16,:,:]+intermediate[16:,:,:]*1j).mean(axis=0)

    PowerSpectrum = np.log10(FFT)
    # rescale
    PowerSpectrum = (PowerSpectrum -PowerSpectrum.min())/(PowerSpectrum.max()-PowerSpectrum.min())*255
    PowerSpectrum = cv2.cvtColor(PowerSpectrum.astype('uint8'),cv2.COLOR_GRAY2BGR)

    if image.flags.writeable is False:
        image = image.copy()

    for box in pred_obj:
        box = box[1:]
        u1,v1 = worldToImage(-box[2],box[1],0)
        u2,v2 = worldToImage(-box[0],box[1],1.6)

        u1 = int(u1/2)
        v1 = int(v1/2)
        u2 = int(u2/2)
        v2 = int(v2/2)

        image = cv2.rectangle(image, (u1,v1), (u2,v2), (0, 0, 255), 3)

    segmapOut_on_image = polarSemantic_image_mapping(polarSegmap=out_seg, image=image, show_fig=None, threshold=threshold, mapping_color=mapping_color, ksize=ksize, dilated_iter=dilated_iter)    # This function projects the segmentation map (polar representation) onto the camera image
    segmapIn_on_image =  polarSemantic_image_mapping(polarSegmap=segmap.squeeze(), image=image,show_fig=None, threshold=threshold, mapping_color=mapping_color, ksize=ksize, dilated_iter=dilated_iter)

    RA_cartesian,_=polarTransform.convertToCartesianImage(np.moveaxis(out_seg,0,1),useMultiThreading=True,
        initialAngle=0, finalAngle=np.pi,order=0,hasColor=False)

    # Make a crop on the angle axis
    RA_cartesian = RA_cartesian[:,256-100:256+100]

    RA_cartesian = np.asarray((RA_cartesian*255).astype('uint8'))
    RA_cartesian = cv2.cvtColor(RA_cartesian, cv2.COLOR_GRAY2BGR)
    RA_cartesian = cv2.resize(RA_cartesian,dsize=(400,512))
    RA_cartesian=cv2.flip(RA_cartesian,flipCode=-1)

    return np.hstack((PowerSpectrum, segmapIn_on_image[:512],segmapOut_on_image[:512],RA_cartesian))

def polarSemantic_image_mapping(polarSegmap, image, show_fig=None, threshold=0.5, mapping_color=[46, 206, 254], ksize=(3,3), dilated_iter=5):
    # This function projects the segmentation map (polar representation) onto the camera image
    # polarSegmap represents the segmentation map in polar coordinates, with axes for radial distance (m) and horizontal azimuth (degree). This function converts it to a top-down view of the lidar or radar point cloud
    # Input:
    #    mapping_color # Target color for binary segmentation (BGR format)
    # ksize: Kernel size for image dilation
    # dilated_iter: Number of iterations for image dilation


    if show_fig is not None:
        plt.figure()
        plt.imshow(polarSegmap)

    # The calculation formulas in the next two lines come from example_dataset.py lines 123-124
    range_segmap = np.arange(polarSegmap.shape[0]) * 2 * 103 / 512  # Get the radial distance (m) in polar coordinates for each pixel in segmap
    azimuth_segmap = (np.arange(polarSegmap.shape[1]) - polarSegmap.shape[1] / 2) * (
        -1) * 0.4  # Get the azimuth angle (degree) in polar coordinates for each pixel in segmap
    # Additionally, the polar-to-Cartesian conversion can also be found in polarTransform.convertToCartesianImage in the DisplayHMI function (line 162 of this py)

    # x and y represent the Cartesian coordinate position of each pixel in the point cloud map, in meters
    x = np.zeros_like(polarSegmap).astype(float)
    y = np.zeros_like(polarSegmap).astype(float)
    for rng_idx in range(polarSegmap.shape[0]):
        for azi_idx in range(polarSegmap.shape[1]):
            y[rng_idx, azi_idx] = range_segmap[rng_idx] * np.cos(azimuth_segmap[azi_idx] / 180 * np.pi)
            x[rng_idx, azi_idx] = range_segmap[rng_idx] * np.sin(azimuth_segmap[azi_idx] / 180 * np.pi)

    segmap_flat = polarSegmap.flatten()
    x_flat = x.flatten()
    y_flat = y.flatten()

    segmap_pc = np.vstack((y_flat, x_flat, segmap_flat)).T
    condition = segmap_pc[:, 2] > threshold  # Threshold to prevent misjudgment of the neural network's out_seg output. In out_seg, non-open road areas are set within the threshold value
    segmap_pc = segmap_pc[condition]

    if show_fig is not None:
        plt.figure()
        plt.plot(-segmap_pc[:, 1], segmap_pc[:, 0], '.')
        plt.xlim(-20, 20)
        plt.ylim(0, 100)
        plt.grid()
    #   plt.savefig('./saved/6-Segmentation_PointCloud.png')

    # Project the segmentation back onto the image
    # Load the camera calibration parameters
    # calib = np.load('/home/root/Documents/FFTRadNet/RadIal/ADCProcessing/DBReader/examples/camera_calib.npy',
    #                 allow_pickle=True).item()  # Load camera calibration parameters, including intrinsic (camera matrix, distribution coefficient) and extrinsic (rotation vector, translation vector)
    camera_height = 0  # Camera position relative to the photo = 0
    segmap_pc_with_zero_height = segmap_pc.copy()
    segmap_pc_with_zero_height[:, [0, 1, 2]] = segmap_pc_with_zero_height[:, [1, 0, 2]]
    segmap_pc_with_zero_height[:, 0] *= -1
    segmap_pc_with_zero_height[:, 2] = camera_height

    # cv2.projectPoints requires input data dimensions to be x, y, z respectively

    # imgpts, _ = cv2.projectPoints(np.array(segmap_pc_with_zero_height),
    #                               calib['extrinsic']['rotation_vector'],
    #                               calib['extrinsic']['translation_vector'],
    #                               calib['intrinsic']['camera_matrix'],
    #                               calib['intrinsic']['distortion_coefficients'])
    imgpts, _ = cv2.projectPoints(np.array(segmap_pc_with_zero_height),
                                  rvecs, tvecs, camera_matrix, dist_coeffs)

    imgpts = imgpts.squeeze(1).astype('int')

    imgpts = (imgpts * 0.5).astype(int)  # Here the image pixel size is no longer 1080*1920 but 540*960, so the imgpts coordinates also need to be divided by 2

    # Keep only points inside the image size
    idx = np.where(
        (imgpts[:, 0] >= 0) & (imgpts[:, 0] < image.shape[1]) & (imgpts[:, 1] >= 0) & (imgpts[:, 1] < image.shape[0]))[
        0]

    valid_pts = imgpts[idx]  # Keep only valid points
    # 3. Image dilation to generate continuous regions (for binary segmentation)
    image_with_segmap = image.copy()
    if len(valid_pts) == 0:
        # No target points, return the original image directly
        return image_with_segmap

    # Step 1: Create an empty mask (single channel)
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Step 2: Mark all target points on the mask (value 255 indicates target)
    mask[valid_pts[:, 1], valid_pts[:, 0]] = 255  # (v, u) corresponds to mask rows and columns

    # Step 3: Define the dilation kernel (controls the shape and size of the region)
    # Larger kernel and more iterations produce more continuous regions (can be adjusted as needed)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)  # Elliptical kernel looks more natural
    dilated_mask = cv2.dilate(mask, kernel, iterations=dilated_iter)  # Dilate for the specified number of iterations

    # Step 4: Convert the single-channel mask to color (matching the target color)
    # mapping_color = [46, 206, 254]  # Target color for binary segmentation (BGR format)
    colored_mask = np.zeros_like(image_with_segmap)
    colored_mask[dilated_mask == 255] = mapping_color  # Color the target area

    # Step 5: Overlay with the original image (semi-transparent effect)
    alpha = 0.5  # Transparency (0-1, higher values make the target more visible)
    image_with_segmap = cv2.addWeighted(
        image_with_segmap,  # Original image
        1 - alpha,  # Original image weight
        colored_mask,  # Dilated target area
        alpha,  # Target area weight
        0  # Brightness offset
    )

    # Optional: Display intermediate results (for debugging)
    # if show_fig is not None:
    #     plt.figure(figsize=(15, 5))
    #     plt.subplot(131)
    #     plt.imshow(mask, cmap='gray')
    #     plt.title("Original Mask (Points)")
    #     plt.subplot(132)
    #     plt.imshow(dilated_mask, cmap='gray')
    #     plt.title("Dilated Mask (Continuous Area)")
    #     plt.subplot(133)
    #     plt.imshow(cv2.cvtColor(image_with_segmap, cv2.COLOR_BGR2RGB))
    #     plt.title("Final Result")
    #     plt.tight_layout()
    #     plt.show()

    return image_with_segmap


    # image_with_segmap = image.copy()
    # image_with_segmap[imgpts[idx, 1], imgpts[idx, 0], :] = [46, 206, 254]  # Set semantic area color [B,G,R] [247, 238, 55]
    #
    # if show_fig is not None:
    #     plt.figure(figsize=(20, 20))
    #     plt.imshow(image)
    #     plt.plot(imgpts[idx, 0], imgpts[idx, 1], 'y.')
    #     plt.show()
    #  #  plt.savefig('./saved/7-Segmentation_Image.png')
    #
    # return image_with_segmap
