import os

import numpy as np
import pickle, copy

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

from mmengine.dataset.base_dataset import  force_full_init

import warnings
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import PixelData

from mmseg.registry import TRANSFORMS
from mmseg.structures import SegDataSample


@COLLATE_FUNCTIONS.register_module()
def RADIal_collate(batch):
    images = []
    FFTs = []
    segmaps = []
    labels = []
    encoded_label = []
    for radar_FFT, segmap, out_label,box_labels,image in batch:
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
            self.decoder = enc.decode
        self.perform_FFT = perform_FFT

        # print("self.perform_FFT=",self.perform_FFT)
        # print("self.root_dir=", self.root_dir)
        self.labels = pd.read_csv(os.path.join(data_root, 'labels.csv')).to_numpy()
        # print(self.labels[0])
        self.labels = self.__remove_nonexist_file__()# Generate self.labels for train/val/test respectively
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
        labels = self.labels[:, 0]
        nonexist_file_ = []
        for i in range(labels.__len__()):
            if self.perform_FFT == 'Custom_RD' or self.perform_FFT == 'ADC':
                curr_file = os.path.join(self.root_dir, 'ADC_Data', "adc_{:06d}.npy".format(labels[i]))
                curr_file_2 = os.path.join(self.data_prefix['img_path'], "adc_{:06d}.npy".format(labels[i]))# The value of self.data_prefix['img_path'] changes depending on whether it is training or validation
            elif self.perform_FFT == 'Read_fft_Data':
                curr_file = os.path.join(self.root_dir, 'radar_FFT', "fft_{:06d}.npy".format(labels[i]))
                curr_file_2 = os.path.join(self.data_prefix['img_path'], "fft_{:06d}.npy".format(labels[i]))
            else:
                raise NotImplementedError("This branch is not yet implemented")

            search_result = os.path.isfile(curr_file) and os.path.isfile( curr_file_2 )
            if (search_result == False):
                nonexist_file_.append(labels[i])
        self.nonexist_file = np.array(nonexist_file_)
        for item in self.nonexist_file:
            index = np.where(self.labels[:, 0] == item)
            self.labels = np.delete(self.labels, index, axis=0)
        return self.labels


    @force_full_init# Override /home/root/Documents/RADIal_VMamba/mmengine/dataset/base_dataset.py get_data_info
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index and automatically call ``full_init`` if the
        dataset has not been fully initialized.

        Args:
            idx (int): The index of data.

        Returns:
            dict: The idx-th annotation of the dataset.
        """
        if self.serialize_data:
            start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
            end_addr = self.data_address[idx].item()
            bytes = memoryview(
                self.data_bytes[start_addr:end_addr])  # type: ignore
            data_info = pickle.loads(bytes)  # type: ignore
        else:
            data_info = copy.deepcopy(self.data_list[idx])
        # Some codebase needs `sample_idx` of data information. Here we convert
        # the idx to a positive number and save it in data information.
        # if idx >= 0:
        #     data_info['sample_idx'] = idx
        # else:
        #     data_info['sample_idx'] = len(self) + idx

        data_info['sample_id'] = copy.deepcopy(self.sample_keys[idx])

        data_info['sample_keys'] = copy.deepcopy(self.sample_keys)
        data_info['label_dict'] = copy.deepcopy(self.label_dict)
        data_info['labels'] = copy.deepcopy(self.labels)
        data_info['encoder'] = copy.deepcopy(self.encoder)
        data_info['decoder'] = copy.deepcopy(self.decoder)
        data_info['perform_FFT']  = copy.deepcopy(self.perform_FFT)
        data_info['data_prefix'] = copy.deepcopy(self.data_prefix)
        data_info['range_fft_coef'] = copy.deepcopy(self.range_fft_coef)
        data_info['numSamplePerChirp'] = copy.deepcopy(self.numSamplePerChirp)
        data_info['doppler_fft_coef'] = copy.deepcopy(self.doppler_fft_coef)
        data_info['numChirps'] = copy.deepcopy(self.numChirps)
        data_info['statistics'] = copy.deepcopy(self.statistics)
        data_info['root_dir'] = copy.deepcopy(self.root_dir)
        data_info['crop'] = copy.deepcopy(self.crop)
        data_info['resize'] = copy.deepcopy(self.resize)
        return data_info

    '''

    def __getitem__(self, index):# Override __getitem__ function
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

    '''






from mmcv.transforms.base import BaseTransform
from mmcv.transforms.builder import TRANSFORMS
@TRANSFORMS.register_module()
class LoadADCDataAndAnnoFromFile(BaseTransform):
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
    def __init__(self,
                 **kwargs
                 ) -> None:
        pass
    def transform(self, results: dict):
        """Functions to load adc data.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

                The contents of the ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:
                        - ``img_path``: filename of the image
                        - ``ori_shape``: original shape of the image as a tuple (h, w, c)
                        - ``img_shape``: shape of the image input to the network as a tuple \
                            (h, w, c).  Note that images may be zero padded on the \
                            bottom/right if the batch tensor is larger than this shape.
                        - ``pad_shape``: shape of padded images
                        - ``scale_factor``: a float indicating the preprocessing scale
                        - ``flip``: a boolean indicating if image flip transform was used
                        - ``flip_direction``: the flipping direction
        Returns:
            dict: The dict contains loaded image and meta information.
        """

        # Get the sample id
        sample_id = results['sample_id']

        # From the sample id, retrieve all the labels ids
        entries_indexes = results['label_dict'][sample_id]

        # Get the objects labels
        box_labels = results['labels'][entries_indexes]

        # Labels contains following parameters:
        # x1_pix	y1_pix	x2_pix	y2_pix	laser_X_m	laser_Y_m	laser_Z_m radar_X_m	radar_Y_m	radar_R_m

        # format as following [Range, Angle, Doppler,laser_X_m,laser_Y_m,laser_Z_m,x1_pix,y1_pix,x2_pix	,y2_pix]
        box_labels = box_labels[:, [10, 11, 12, 5, 6, 7, 1, 2, 3, 4]].astype(np.float32)    # The values of x1_pix, y1_pix, x2_pix, y2_pix divided by 2 give the correct coordinates in the image

        ######################
        #  Encode the labels #
        ######################
        out_label = []
        if (results['encoder'] != None):
            out_label = results['encoder'](box_labels).copy()

        # Read the Radar data
        if results['perform_FFT'] == 'Custom_RD':
            # If you want to perform custom operations on the radar.
            # Here we include an fftshift to center zero doppler frequency
            # For a rectangular window remove multiplication of coefs
            # I recommend saving them with the commented code at the bottom
            # of the if statement and running 'dataset/print_dataset_statistcs.py'
            # This will give you normalization values and save to the folder you have created under the root_dir
            raise NotImplementedError("This branch is not yet verified")
            radar_name = os.path.join(results['data_prefix']['img_path'], "adc_{:06d}.npy".format(sample_id))
            complex_adc = np.load(radar_name, allow_pickle=True)
            complex_adc = complex_adc - np.mean(complex_adc, axis=(0, 1))
            range_fft = mkl_fft.fft(np.multiply(complex_adc, results['range_fft_coef']), results['numSamplePerChirp'], axis=0)
            input = mkl_fft.fft(np.multiply(range_fft, results['doppler_fft_coef']), results['numChirps'], axis=1)
            # Shift doppler zero freqency bin to center of spectrum
            radar_FFT = np.fft.fftshift(input, axes=1).astype(np.complex64)
            radar_FFT = np.concatenate([input.real, input.imag], axis=2)

            if (results['statistics'] is not None):
                for i in range(len(results['statistics']['input_mean'])):
                    radar_FFT[..., i] -= results['statistics']['input_mean'][i]
                    radar_FFT[..., i] /= results['statistics']['input_std'][i]

            # out_name = os.path.join(results['root_dir'],'RD_Shift',"fft_{:06d}.npy".format(sample_id))
            # np.save(out_name,radar_FFT)

        elif results['perform_FFT'] == 'ADC':
            # Utilize Raw ADC
            radar_name = os.path.join(results['data_prefix']['img_path'], "adc_{:06d}.npy".format(sample_id))
            radar_FFT = np.load(radar_name, allow_pickle=True)

        elif results['perform_FFT'] == 'Read_fft_Data':
            # Default method. Uses precomputed FFT's.
            radar_name = os.path.join(results['data_prefix']['img_path'], "fft_{:06d}.npy".format(sample_id))
            input = np.load(radar_name, allow_pickle=True)
            radar_FFT = np.concatenate([input.real, input.imag], axis=2)
            if (results['statistics'] is not None):
                for i in range(len(results['statistics']['input_mean'])):
                    radar_FFT[..., i] -= results['statistics']['input_mean'][i]
                    radar_FFT[..., i] /= results['statistics']['input_std'][i]
        else:
            raise NotImplementedError(f"The current perform_FFT field (={results['perform_FFT']}) is not supported" )

        assert radar_name==results['img_path']
        # Read the segmentation map
        segmap_name = os.path.join(results['data_prefix']['seg_map_path'], "freespace_{:06d}.png".format(sample_id))
        # assert segmap_name == results['seg_map_path']

        segmap = Image.open(segmap_name)  # [512,900]
        ori_seg_shape  = segmap.size
        # 512 pix for the range and 900 pix for the horizontal FOV (180deg)
        # We crop the fov to 89.6deg
        segmap = results['crop'](segmap)
        # and we resize to half of its size
        segmap = np.asarray(results['resize'](segmap)) == 255
        # Read the camera image
        img_name = os.path.join(results['root_dir'], 'camera', "image_{:06d}.jpg".format(sample_id))
        image = np.asarray(Image.open(img_name))
        # output = {
        #     'radar_FFT': radar_FFT,
        #     'segmap': segmap,
        #     'out_label': out_label,
        #     'box_labels': box_labels,
        #     'image': image
        # }

        meta_iofo = {
            "img_path" : results['img_path'],
            "ori_shape" :  segmap.shape,#[256,224]   #image.shape[0:2], #radar_FFT.shape=[512,256,16],
            "img_shape" :  segmap.shape,#image.shape[0:2], #radar_FFT.shape,
            "seg_map_path" : results['seg_map_path'],
            "seg_shape" : segmap.shape,#[256,224]
            "ori_seg_shape": ori_seg_shape,#[900,512]
            "camera_path":img_name,
            "camera_shape": image.shape,#[540,960,3]
        }

        output = (
            radar_FFT,
            segmap,
            out_label,
            box_labels,
            image,
            meta_iofo
        )
        # return radar_FFT, segmap, out_label, box_labels, image
        return output




@TRANSFORMS.register_module()
class PackRadialSegInputs(BaseTransform):
    """Pack the inputs data for the semantic segmentation.

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_path``: filename of the image

        - ``ori_shape``: original shape of the image as a tuple (h, w, c)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``pad_shape``: shape of padded images

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be packed from
            ``SegDataSample`` and collected in ``data[img_metas]``.
            Default: ``('img_path', 'ori_shape',
            'img_shape', 'pad_shape', 'scale_factor', 'flip',
            'flip_direction')``
    """
    def __init__(self, data_mode,
                 meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'reduce_zero_label')
                 ):
        self.meta_keys = meta_keys
        self.data_mode = data_mode

    def transform(self, results: tuple) -> dict:
        """Method to pack the Radial input data.
        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.

            - 'data_sample' (obj:`SegDataSample`): The annotation info (segmentation) of the sample.
                            (obj:`LabelDataSample`): The detection label info (label) of the sample.
        """
        packed_results = dict()

        # Add adc_data
        if self.data_mode == 'ADC':
            radar_FFT = torch.from_numpy(results[0]).to(torch.complex64)
            radar_FFT =  radar_FFT.permute(2, 0, 1)
        elif self.data_mode == 'Read_fft_Data':
            radar_FFT = torch.from_numpy(results[0]).to(torch.complex64)
            radar_FFT = radar_FFT.permute(2, 0, 1)
        else:
            raise NotImplementedError(f"Data mode {self.data_mode} not implemented. Refer to line 152 of FFRRadNet.1-Train.py for modification!")

        packed_results['inputs'] = radar_FFT


        # Add radar_freespace
        data_sample = SegDataSample()
        gt_sem_seg_data  = dict( data=torch.from_numpy( np.expand_dims(results[1],axis=0) ).double() )
        data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)
        # Add encoded_label
        gt_encoded_label_data  = dict( data=torch.from_numpy(results[2]).double() )
        data_sample.set_data(dict(gt_label_map=MultiChannelData(**gt_encoded_label_data)))
        # Add img meta info
        img_meta = results[5]
        data_sample.set_metainfo(img_meta)

        packed_results['data_samples'] = data_sample


        # Add box_labels
        # packed_results['box_labels'] = torch.from_numpy(results[3])
        gt_box_label_data = dict(data=torch.from_numpy(results[3].copy()))
        data_sample.set_data(dict(gt_box_label=MultiChannelData(**gt_box_label_data)))

        # Add camera image
        # packed_results['image'] = torch.tensor(results[4])
        gt_image = dict(data=torch.from_numpy(results[4].copy()))
        data_sample.set_data(dict(gt_image=MultiChannelData(**gt_image)))

        return packed_results

# Add MultiChannelData data structure support to PackRadialSegInputs for storing gt_label_data
from mmengine.structures import BaseDataElement
class MultiChannelData(BaseDataElement):
    # Multi-channel data structure for storing radar labels in [C,H,W] format (i.e., gt_label_data)
    def __init__(self, data=None, metainfo=None):
        super().__init__()
        if data is not None:
            self.data = data
        if metainfo is not None:
            self.set_metainfo( metainfo )
    @property
    def data(self):
        """Get multi-channel data"""
        return self._data

    @data.setter    # Set multi-channel data, ensure it is of type torch.Tensor
    def data(self, value):
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        self.set_field(value, '_data', dtype=torch.Tensor)

    @property
    def shape(self):
        return self.data.shape if hasattr(self, '_data') else None
    def to(self,device, non_blocking=False):
        self.data = self.data.to(device, non_blocking=non_blocking)
        return self
    def cuda(self, device=None):
        self.data = self.data.cuda(device)
        return self
    def cpu(self):
        self.data = self.data.cpu()
        return self
    def detach(self):
        self.data = self.data.detach()
        return self

class SegDataSample(SegDataSample):
    """Extend SegDataSample to support multi-channel data"""
    @property
    def multi_channel_data(selfself) -> MultiChannelData:
        """Get multi-channel data attribute"""
        return self._multi_channel_data
    @multi_channel_data.setter
    def multi_channel_data(self, value:MultiChannelData)->None:
        """Set multi-channel data attribute"""
        if not isinstance(value, MultiChannelData):
            # If raw data is passed in, automatically wrap it as MultiChannelData
            if isinstance(value, torch.Tensor):
                data = value
            else:
                data = torch.tensor(value)
            value = MultiChannelData(data = data)
        self.set_field(value, '_multi_channel_data', dtype=MultiChannelData)
    @multi_channel_data.deleter
    def multi_channel_data(self) -> None:
        """Delete multi-channel data attribute"""
        del self._multi_channel_data









import math
from typing import Mapping, Optional, Sequence, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import MODELS
from mmengine.structures import BaseDataElement
from mmengine.utils import is_seq_of
from mmengine.model.utils import stack_batch
from mmengine.model.base_model.data_preprocessor import BaseDataPreprocessor
@MODELS.register_module()
class RadialDataPreprocessor(BaseDataPreprocessor):
    """Radial data pre-processor.

    Accepts the data sampled by the dataloader, and preprocesses it into the
    format of the models input. ``RadialDataPreprocessor`` provides the
    basic data pre-processing as follows

    - Collates and moves data to the target device.
    - Normalizes adc_data with defined std and mean if necessary.
    - Stack inputs to batch_inputs.

    For ``RadialDataPreprocessor``, the dimension of the single inputs must be
    (16, H, W).

    Note:
        ``RadialDataPreprocessor`` and its subclass is built in the
        constructor of :class:`BaseDataset`.

    Args:
        mean (Sequence[complex64], optional): The pixel mean of adc_data
            channels. If the length of `mean` is 1, it means all
            channels have the same mean value.
            If it is not specified, adc_data will not be normalized. Defaults
            None.
        std (Sequence[complex64], optional): The pixel standard deviation of
            image channels. If the length of `std` is 1,
            it means all channels have the same standard deviation, or the
            input is a gray image.  If it is not specified, adc_data will
            not be normalized. Defaults None.
        non_blocking (bool): Whether block current process
            when transferring data to device.
            New in version v0.3.0.

    Note:
        if adc_data do not need to be normalized, `std` and `mean` should be
        both set to None, otherwise both of them should be set to a tuple of
        corresponding values.
    """

    def __init__(self,
                 mean: Optional[Sequence[Union[complex]]] = None,
                 std: Optional[Sequence[Union[complex]]] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 non_blocking: Optional[bool] = False):
        super().__init__(non_blocking)

        assert (mean is None) == (std is None), (
            'mean and std should be both None or tuple')
        if mean is not None:
            assert len(mean) == 16 or len(mean) == 1, (
                '`mean` should have 1 or 16 values, to be compatible with '
                f'Radial adc_data, but got {len(mean)} values')
            assert len(std) == 16 or len(std) == 1, (  # type: ignore
                '`std` should have 1 or 16 values, to be compatible with '  # type: ignore # noqa: E501
                f'Radial adc_data, but got {len(std)} values')  # type: ignore
            self._enable_normalize = True
            self.register_buffer('mean',
                                 torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std',
                                 torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value

    def forward(self, data: dict, training: bool = False) -> Union[dict, list]:
        """Performs normalization、padding based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): Data sampled from dataset. If the collate
                function of DataLoader is :obj:`pseudo_collate`, data will be a
                list of dict. If collate function is :obj:`default_collate`,
                data will be a tuple with batch input tensor and list of data
                samples.
            training (bool): Whether to enable training time augmentation. If
                subclasses override this method, they can perform different
                preprocessing strategies for training and testing based on the
                value of ``training``.

        Returns:
            dict or list: Data in the same format as the models input.
        """
        data = self.cast_data(data)  # type: ignore
        _batch_inputs = data['inputs']
        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_inputs = []
            for _batch_input in _batch_inputs:
                # Convert to float after channel conversion to ensure
                # efficiency
                assert _batch_input.dtype== torch.complex64, "adc data type should be complex64"
                # Normalization.
                if self._enable_normalize:
                    if self.mean.shape[0] == 16:
                        assert _batch_input.dim(
                        ) == 3 and _batch_input.shape[0] == 16, (
                            'If the mean has 16 values, the input tensor '
                            'should in shape of (16, H, W), but got the tensor '
                            f'with shape {_batch_input.shape}')
                    _batch_input = (_batch_input - self.mean) / self.std
                batch_inputs.append(_batch_input)
            # Pad and stack Tensor.
            batch_inputs = stack_batch(batch_inputs, self.pad_size_divisor,
                                       self.pad_value)
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, torch.Tensor):
            raise NotImplementedError("This branch has not been verified yet")
            assert _batch_inputs.dim() == 4, (
                'The input of `RadialDataPreprocessor` should be a NCHW tensor '
                'or a list of tensor, but got a tensor with shape: '
                f'{_batch_inputs.shape}')

            # Convert to float after channel conversion to ensure
            # efficiency
            assert _batch_inputs.dtype == torch.complex64, "adc data type should be complex64"

            if self._enable_normalize:
                _batch_inputs = (_batch_inputs - self.mean) / self.std
            h, w = _batch_inputs.shape[2:]
            target_h = math.ceil(
                h / self.pad_size_divisor) * self.pad_size_divisor
            target_w = math.ceil(
                w / self.pad_size_divisor) * self.pad_size_divisor
            pad_h = target_h - h
            pad_w = target_w - w
            batch_inputs = F.pad(_batch_inputs, (0, pad_w, 0, pad_h),
                                 'constant', self.pad_value)
        else:
            raise TypeError('Output of `cast_data` should be a dict of '
                            'list/tuple with inputs and data_samples, '
                            f'but got {type(data)}: {data}')
        data['inputs'] = batch_inputs
        data.setdefault('data_samples', None)
        return data
