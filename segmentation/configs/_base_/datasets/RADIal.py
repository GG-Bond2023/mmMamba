# Tutorial: https://mmsegmentation.readthedocs.io/zh-cn/latest/user_guides/1_config.html
# dataset settings

import os

dataset_type = 'RADIalDataset'
# data_root = 'segmentation/data/RADIal'
data_root = '/code/datasets/RADIal'

data_mode = 'ADC'

img_path = 'ADC_Data'
seg_map_path = 'radar_Freespace'
img_suffix = '.npy'
seg_map_suffix = '.png'

geometry={
    "ranges": [512, 896, 1],
    "resolution": [0.201171875, 0.2],
    "size": 3
}

statistics = {
    "input_mean": [-2.6244e-03, -2.1335e-01, 1.8789e-02, -1.4427e+00, -3.7618e-01,
                   1.3594e+00, -2.2987e-01, 1.2244e-01, 1.7359e+00, -6.5345e-01,
                   3.7976e-01, 5.5521e+00, 7.7462e-01, -1.5589e+00, -7.2473e-01,
                   1.5182e+00, -3.7189e-01, -8.8332e-02, -1.6194e-01, 1.0984e+00,
                   9.9929e-01, -1.0495e+00, 1.9972e+00, 9.2869e-01, 1.8991e+00,
                   -2.3772e-01, 2.0000e+00, 7.7737e-01, 1.3239e+00, 1.1817e+00,
                   -6.9696e-01, 4.4288e-01],
    "input_std": [20775.3809, 23085.5000, 23017.6387, 14548.6357, 32133.5547, 28838.8047,
                  27195.8945, 33103.7148, 32181.5273, 35022.1797, 31259.1895, 36684.6133,
                  33552.9258, 25958.7539, 29532.6230, 32646.8984, 20728.3320, 23160.8828,
                  23069.0449, 14915.9053, 32149.6172, 28958.5840, 27210.8652, 33005.6602,
                  31905.9336, 35124.9180, 31258.4316, 31086.0273, 33628.5352, 25950.2363,
                  29445.2598, 32885.7422],
    "reg_mean": [0.4048094369863972, 0.3997392847799934],
    "reg_std": [0.6968599580482511, 0.6942950877813826]
}

# crop_size = (512, 512)# Crop size for training

train_pipeline = [# The __getitem__ function in mmengine/dataset/base_dataset.py calls self.prepare_data(idx) which invokes this pipeline sequentially. However, the newly added RADIal_dataset_process.py overrides __getitem__ to directly complete the adc_data and annotations loading process.
    dict(type='LoadADCDataAndAnnoFromFile'),
    # dict(type='LoadImageFromFile'), # Step 1, load image from file path
    # dict(type='LoadRADIalFromFile'), # Step 1, load RADIal data from file path
    # dict(type='LoadAnnotations', reduce_zero_label=True), # Step 2, load annotation image for current image
    #
    # # dict(
    # #     type='RandomResize',# Data augmentation pipeline: resize input image and its annotations
    # #     scale=(2048, 512),# Image crop size
    # #     ratio_range=(0.5, 2.0),# Data augmentation ratio range
    # #     keep_ratio=True),# Whether to keep aspect ratio when resizing
    # # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    # # dict(type='RandomFlip', prob=0.5),
    # # dict(type='PhotoMetricDistortion'),
    # dict(type='PackSegInputs')
    dict(type='PackRadialSegInputs', data_mode=data_mode)

]
test_pipeline = [
    dict(type='LoadADCDataAndAnnoFromFile'),
    # dict(type='LoadImageFromFile'),
    # dict(type='LoadRADIalFromFile'),  # Step 1, load RADIal data from file path
    # dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    # dict(type='LoadAnnotations', reduce_zero_label=True),
    # dict(type='PackSegInputs')
    dict(type='PackRadialSegInputs', data_mode=data_mode)
]

tta_pipeline = [
    dict(type='LoadADCDataAndAnnoFromFile'),
    # dict(type='LoadImageFromFile', backend_args=None),
    # dict(type='LoadRADIalFromFile'), # Step 1, load RADIal data from file path
    dict(
        type='TestTimeAug',
        transforms=[
            # [
            #     dict(type='Resize', scale_factor=r, keep_ratio=True)
            #     for r in img_ratios
            # ],
            # [
            #     dict(type='RandomFlip', prob=0., direction='horizontal'),
            #     dict(type='RandomFlip', prob=1., direction='horizontal')
            # ],
            # [dict(type='LoadAnnotations')],
            # [dict(type='PackSegInputs')],
            [dict(type='PackRadialSegInputs', data_mode=data_mode)]
        ]
    )
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,# Close worker processes after each epoch to speed up training
    sampler=dict(type='InfiniteSampler', shuffle=True),# Random shuffle during training
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        statistics = statistics,
        geometry = geometry,
        encoder = True,
        data_prefix=dict(img_path=os.path.join( img_path,'train'), seg_map_path=os.path.join(seg_map_path, 'train') ),
        difficult = True,
        perform_FFT=data_mode,
        pipeline=train_pipeline
    ),
    # collate_fn=dict(type='RADIal_collate')
)
val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        statistics = statistics,
        geometry = geometry,
        encoder= True,
        difficult=True,
        perform_FFT=data_mode,
        data_prefix=dict(img_path=os.path.join( img_path,'val'), seg_map_path=os.path.join(seg_map_path, 'val') ),
        pipeline=test_pipeline
    ),
    # collate_fn=dict(type='RADIal_collate')
)
test_dataloader = val_dataloader
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        statistics=statistics,
        geometry=geometry,
        encoder=True,
        difficult=True,
        perform_FFT=data_mode,
        data_prefix=dict(img_path=os.path.join(img_path, 'test'), seg_map_path=os.path.join(seg_map_path, 'test') ),
        pipeline=test_pipeline,
    ),
)

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator