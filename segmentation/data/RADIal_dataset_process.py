import os

import numpy as np

from torchvision.transforms import Resize,CenterCrop
import torchvision.transforms as transform
import pandas as pd
import math
from PIL import Image
import torch
from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset
# from mmseg.datasets.custom import CustomDataset
from segmentation.data.ra_encoder import ra_encoder

from mmengine.dataset import COLLATE_FUNCTIONS
@COLLATE_FUNCTIONS.register_module()
def RADIal_collate(batch):
    images = []
    FFTs = []
    segmaps = []
    labels = []
    encoded_label = []
    for radar_FFT, segmap,out_label,box_labels,image in batch:
        FFTs.append(torch.tensor(radar_FFT).permute(2,0,1))
        segmaps.append(torch.tensor(segmap))
        encoded_label.append(torch.tensor(out_label))
        images.append(torch.tensor(image))
        labels.append(torch.from_numpy(box_labels))
    return torch.stack(FFTs), torch.stack(encoded_label),torch.stack(segmaps),labels,torch.stack(images)


@DATASETS.register_module()
class RADIalDataset(BaseSegDataset):
    # METAINFO = dict(
    #     classes=('Background', 'OpenRoad'),
    #     palette=[[0, 0, 0], [255, 255, 255]])
    def __init__(self, data_root, img_suffix, seg_map_suffix, statistics=None, geometry=None, encoder=False, difficult=False, perform_FFT='Default', **kwargs):

        super().__init__(
            data_root = data_root,
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs
        )
        self.root_dir = data_root
        self.statistics = statistics
        self.geometry = geometry

        if encoder == True:
            enc = ra_encoder(geometry=self.geometry,
                             statistics=self.statistics,
                             regression_layer=2)
            self.encoder = enc.encode

        self.perform_FFT = perform_FFT

        # print("self.perform_FFT=",self.perform_FFT)
        # print("self.root_dir=", self.root_dir)
        self.labels = pd.read_csv(os.path.join(data_root, 'labels.csv')).to_numpy()
        # print(self.labels[0])
        self.labels = self.__remove_nonexist_file__()
        self.numChirps = 256
        self.numSamplePerChirp = 512
        self.numRxAnt = 16
        self.numTxAnt = 12
        hanningWindowRange = (0.54 - 0.46 * np.cos(
            ((2 * math.pi * np.arange(self.numSamplePerChirp)) / (self.numSamplePerChirp - 1))))
        hanningWindowDoppler = (
                    0.54 - 0.46 * np.cos(((2 * math.pi * np.arange(self.numChirps)) / (self.numChirps - 1))))
        hanningWindowAzimuth = (0.54 - 0.46 * np.cos(((2 * math.pi * np.arange(self.numRxAnt)) / (self.numRxAnt - 1))))
        self.range_fft_coef = np.expand_dims(
            np.repeat(np.expand_dims(hanningWindowRange, 1), repeats=self.numChirps, axis=1), 2)
        self.doppler_fft_coef = np.expand_dims(
            np.repeat(np.expand_dims(hanningWindowDoppler, 1).transpose(), repeats=self.numSamplePerChirp, axis=0), 2)

        # Keeps only easy samples
        if (difficult == False):
            ids_filters = []
            ids = np.where(self.labels[:, -1] == 0)[0]
            ids_filters.append(ids)
            ids_filters = np.unique(np.concatenate(ids_filters))
            self.labels = self.labels[ids_filters]

        # Gather each input entries by their sample id
        self.unique_ids = np.unique(self.labels[:, 0])
        self.label_dict = {}
        for i, ids in enumerate(self.unique_ids):
            sample_ids = np.where(self.labels[:, 0] == ids)[0]
            self.label_dict[ids] = sample_ids
        self.sample_keys = list(self.label_dict.keys())

        self.resize = Resize((256, 224), interpolation=transform.InterpolationMode.NEAREST)
        self.crop = CenterCrop((512, 448))

    def __remove_nonexist_file__(self):
        print("*****************************")
        labels = self.labels[:, 0]
        nonexist_file_ = []
        for i in range(labels.__len__()):
            if self.perform_FFT == 'Custom_RD' or self.perform_FFT == 'ADC':
                curr_file = os.path.join(self.root_dir, 'ADC_Data', "adc_{:06d}.npy".format(labels[i]))
            else:
                curr_file = os.path.join(self.root_dir, 'radar_FFT', "fft_{:06d}.npy".format(labels[i]))
            search_result = os.path.isfile(curr_file)
            if (search_result == False):
                nonexist_file_.append(labels[i])
        self.nonexist_file = np.array(nonexist_file_)
        for item in self.nonexist_file:
            index = np.where(self.labels[:, 0] == item)
            self.labels = np.delete(self.labels, index, axis=0)
        return self.labels

    def __len__(self):
        return len(self.label_dict)

    #
    # def load_data_list(self, **kwargs) -> List[dict]:
    #     """Load annotation from directory or annotation file.
    #
    #     Returns:
    #         list[dict]: All data info of dataset.
    #     """
    #     data_list = []
    #     img_dir = self.data_prefix.get('img_path', None)
    #     ann_dir = self.data_prefix.get('seg_map_path', None)
    #     if not osp.isdir(self.ann_file) and self.ann_file:
    #         assert osp.isfile(self.ann_file), \
    #             f'Failed to load `ann_file` {self.ann_file}'
    #         lines = mmengine.list_from_file(
    #             self.ann_file, backend_args=self.backend_args)
    #         for line in lines:
    #             img_name = line.strip()
    #             data_info = dict(
    #                 img_path=osp.join(img_dir, img_name + self.img_suffix))
    #             if ann_dir is not None:
    #                 seg_map = img_name + self.seg_map_suffix
    #                 data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
    #             data_info['label_map'] = self.label_map
    #             data_info['reduce_zero_label'] = self.reduce_zero_label
    #             data_info['seg_fields'] = []
    #             data_list.append(data_info)
    #     else:
    #         _suffix_len = len(self.img_suffix)
    #         for img in fileio.list_dir_or_file(
    #                 dir_path=img_dir,
    #                 list_dir=False,
    #                 suffix=self.img_suffix,
    #                 recursive=True,
    #                 backend_args=self.backend_args):
    #             data_info = dict(img_path=osp.join(img_dir, img))
    #             if ann_dir is not None:
    #                 seg_map = img[:-_suffix_len] + self.seg_map_suffix
    #                 data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
    #             data_info['label_map'] = self.label_map
    #             data_info['reduce_zero_label'] = self.reduce_zero_label
    #             data_info['seg_fields'] = []
    #             data_list.append(data_info)
    #         data_list = sorted(data_list, key=lambda x: x['img_path'])
    #     return data_list


    def __getitem__(self, index):#重写__getitem__函数
        # Get the sample id
        sample_id = self.sample_keys[index]

        # From the sample id, retrieve all the labels ids
        entries_indexes = self.label_dict[sample_id]

        # Get the objects labels
        box_labels = self.labels[entries_indexes]

        # Labels contains following parameters:
        # x1_pix	y1_pix	x2_pix	y2_pix	laser_X_m	laser_Y_m	laser_Z_m radar_X_m	radar_Y_m	radar_R_m

        # format as following [Range, Angle, Doppler,laser_X_m,laser_Y_m,laser_Z_m,x1_pix,y1_pix,x2_pix	,y2_pix]
        box_labels = box_labels[:, [10, 11, 12, 5, 6, 7, 1, 2, 3, 4]].astype(np.float32)

        ######################
        #  Encode the labels #
        ######################
        out_label = []
        if (self.encoder != None):
            out_label = self.encoder(box_labels).copy()

        # Read the Radar data
        if self.perform_FFT == 'Custom_RD':
            # If you want to perform custom operations on the radar.
            # Here we include an fftshift to center zero doppler frequency
            # For a rectangular window remove multiplication of coefs
            # I recommend saving them with the commented code at the bottom
            # of the if statement and running 'dataset/print_dataset_statistcs.py'
            # This will give you normalization values and save to the folder you have created under the root_dir
            radar_name = os.path.join(self.data_prefix['img_path'], "adc_{:06d}.npy".format(sample_id))
            complex_adc = np.load(radar_name, allow_pickle=True)
            complex_adc = complex_adc - np.mean(complex_adc, axis=(0, 1))
            range_fft = mkl_fft.fft(np.multiply(complex_adc, self.range_fft_coef), self.numSamplePerChirp, axis=0)
            input = mkl_fft.fft(np.multiply(range_fft, self.doppler_fft_coef), self.numChirps, axis=1)
            # Shift doppler zero freqency bin to center of spectrum
            radar_FFT = np.fft.fftshift(input, axes=1).astype(np.complex64)
            radar_FFT = np.concatenate([input.real, input.imag], axis=2)

            if (self.statistics is not None):
                for i in range(len(self.statistics['input_mean'])):
                    radar_FFT[..., i] -= self.statistics['input_mean'][i]
                    radar_FFT[..., i] /= self.statistics['input_std'][i]

            # out_name = os.path.join(self.root_dir,'RD_Shift',"fft_{:06d}.npy".format(sample_id))
            # np.save(out_name,radar_FFT)

        elif self.perform_FFT == 'ADC':
            # Utilize Raw ADC
            radar_name = os.path.join(self.data_prefix['img_path'], "adc_{:06d}.npy".format(sample_id))
            radar_FFT = np.load(radar_name, allow_pickle=True)

        else:
            # Default method. Uses precomputed FFT's.
            radar_name = os.path.join(self.data_prefix['img_path'], "fft_{:06d}.npy".format(sample_id))
            input = np.load(radar_name, allow_pickle=True)
            radar_FFT = np.concatenate([input.real, input.imag], axis=2)
            if (self.statistics is not None):
                for i in range(len(self.statistics['input_mean'])):
                    radar_FFT[..., i] -= self.statistics['input_mean'][i]
                    radar_FFT[..., i] /= self.statistics['input_std'][i]

        # Read the segmentation map
        segmap_name = os.path.join(self.data_prefix['seg_map_path'], "freespace_{:06d}.png".format(sample_id))
        segmap = Image.open(segmap_name)  # [512,900]
        # 512 pix for the range and 900 pix for the horizontal FOV (180deg)
        # We crop the fov to 89.6deg
        segmap = self.crop(segmap)
        # and we resize to half of its size
        segmap = np.asarray(self.resize(segmap)) == 255

        # Read the camera image
        img_name = os.path.join(self.root_dir, 'camera', "image_{:06d}.jpg".format(sample_id))
        image = np.asarray(Image.open(img_name))

        return radar_FFT, segmap, out_label, box_labels, image













import warnings
from typing import Optional

import mmengine.fileio as fileio
import numpy as np

import mmcv
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.builder import TRANSFORMS

@TRANSFORMS.register_module()
class LoadADCDataFromFile(BaseTransform):
    """Load ADC Data from file.

    Required Keys:

    - adc_data_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv.imfrombytes`.
            See :func:`mmcv.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """
    def transform(self, results: dict):
        """Functions to load adc data.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename)
                img_bytes = file_client.get(filename)
            else:
                img_bytes = fileio.get(
                    filename, backend_args=self.backend_args)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f'failed to load image: {filename}'
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results


'''
    def __getitem__(self, index):#重写__getitem__函数
        # Get the sample id
        sample_id = self.sample_keys[index]

        # From the sample id, retrieve all the labels ids
        entries_indexes = self.label_dict[sample_id]

        # Get the objects labels
        box_labels = self.labels[entries_indexes]

        # Labels contains following parameters:
        # x1_pix	y1_pix	x2_pix	y2_pix	laser_X_m	laser_Y_m	laser_Z_m radar_X_m	radar_Y_m	radar_R_m

        # format as following [Range, Angle, Doppler,laser_X_m,laser_Y_m,laser_Z_m,x1_pix,y1_pix,x2_pix	,y2_pix]
        box_labels = box_labels[:, [10, 11, 12, 5, 6, 7, 1, 2, 3, 4]].astype(np.float32)

        ######################
        #  Encode the labels #
        ######################
        out_label = []
        if (self.encoder != None):
            out_label = self.encoder(box_labels).copy()

        # Read the Radar data
        if self.perform_FFT == 'Custom_RD':
            # If you want to perform custom operations on the radar.
            # Here we include an fftshift to center zero doppler frequency
            # For a rectangular window remove multiplication of coefs
            # I recommend saving them with the commented code at the bottom
            # of the if statement and running 'dataset/print_dataset_statistcs.py'
            # This will give you normalization values and save to the folder you have created under the root_dir
            radar_name = os.path.join(self.root_dir, 'ADC_Data', "adc_{:06d}.npy".format(sample_id))
            complex_adc = np.load(radar_name, allow_pickle=True)
            complex_adc = complex_adc - np.mean(complex_adc, axis=(0, 1))
            range_fft = mkl_fft.fft(np.multiply(complex_adc, self.range_fft_coef), self.numSamplePerChirp, axis=0)
            input = mkl_fft.fft(np.multiply(range_fft, self.doppler_fft_coef), self.numChirps, axis=1)
            # Shift doppler zero freqency bin to center of spectrum
            radar_FFT = np.fft.fftshift(input, axes=1).astype(np.complex64)
            radar_FFT = np.concatenate([input.real, input.imag], axis=2)

            if (self.statistics is not None):
                for i in range(len(self.statistics['input_mean'])):
                    radar_FFT[..., i] -= self.statistics['input_mean'][i]
                    radar_FFT[..., i] /= self.statistics['input_std'][i]

            # out_name = os.path.join(self.root_dir,'RD_Shift',"fft_{:06d}.npy".format(sample_id))
            # np.save(out_name,radar_FFT)

        elif self.perform_FFT == 'ADC':
            # Utilize Raw ADC
            radar_name = os.path.join(self.root_dir, 'ADC_Data', "adc_{:06d}.npy".format(sample_id))
            radar_FFT = np.load(radar_name, allow_pickle=True)

        else:
            # Default method. Uses precomputed FFT's.
            radar_name = os.path.join(self.root_dir, 'radar_FFT', "fft_{:06d}.npy".format(sample_id))
            input = np.load(radar_name, allow_pickle=True)
            radar_FFT = np.concatenate([input.real, input.imag], axis=2)
            if (self.statistics is not None):
                for i in range(len(self.statistics['input_mean'])):
                    radar_FFT[..., i] -= self.statistics['input_mean'][i]
                    radar_FFT[..., i] /= self.statistics['input_std'][i]

        # Read the segmentation map
        segmap_name = os.path.join(self.root_dir, 'radar_Freespace', "freespace_{:06d}.png".format(sample_id))
        segmap = Image.open(segmap_name)  # [512,900]
        # 512 pix for the range and 900 pix for the horizontal FOV (180deg)
        # We crop the fov to 89.6deg
        segmap = self.crop(segmap)
        # and we resize to half of its size
        segmap = np.asarray(self.resize(segmap)) == 255

        # Read the camera image
        img_name = os.path.join(self.root_dir, 'camera', "image_{:06d}.jpg".format(sample_id))
        image = np.asarray(Image.open(img_name))

        return radar_FFT, segmap, out_label, box_labels, image
'''