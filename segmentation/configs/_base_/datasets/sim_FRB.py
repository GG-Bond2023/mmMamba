# Tutorial: https://mmsegmentation.readthedocs.io/zh-cn/latest/user_guides/1_config.html
# dataset settings
dataset_type = 'simFRBDataset'
data_root = 'data/FAST_FREX/sim_sample'
crop_size = (512, 512)# Crop size for training
# #
# custom_imports = dict(
#     imports = ['data.transforms'],
#     allow_failed_imports = False
# )

train_pipeline = [
    # dict(type='LoadFitsFromFile'), # Step 1, load image from file path
    dict(type='LoadImageFromFile'), # Step 1, load image from file path
    dict(type='LoadAnnotations', reduce_zero_label=False),
    # dict(type='LoadFitsAnnotations'), # Step 2, load annotation image for current image

    # dict(type='sim_FRB_split'),

    dict(
        type='LabelMapper',# Custom label mapping
        mapping = {0:0, 255:1},# Map 255 to 1
        default_val = -100# Unspecified label values mapped to 255
    ),

    dict(
        type='RandomResize',# Data augmentation pipeline: resize input image and its annotations
        scale=(2048, 512),# Image crop size
        ratio_range=(0.5, 2.0),# Data augmentation ratio range
        keep_ratio=True),# Whether to keep aspect ratio when resizing
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='LabelMapper',  # Custom label mapping
        mapping={0: 0, 255: 1},  # Map 255 to 1
        default_val=-100  # Unspecified label values mapped to 255
    ),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='LabelMapper',mapping = {0:0, 255:1},default_val = -100)],[dict(type='PackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,# Close worker processes after each epoch to speed up training
    sampler=dict(type='InfiniteSampler', shuffle=True),# Random shuffle during training
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train/data', seg_map_path='train/label'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='eval/data',
            seg_map_path='eval/label'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])# Accuracy evaluation method, using IoUMetric here
test_evaluator = val_evaluator