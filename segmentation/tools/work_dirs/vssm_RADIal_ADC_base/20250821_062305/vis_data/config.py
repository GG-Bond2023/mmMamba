backbone_norm_cfg = dict(requires_grad=True, type='LN')
data_mode = 'ADC'
data_preprocessor = dict(mean=None, std=None, type='RadialDataPreprocessor')
data_root = 'segmentation/data/RADIal'
dataset_type = 'RADIalDataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=10000, type='CheckpointHook'),
    logger=dict(interval=100, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        dilated_iter=5,
        draw=True,
        interval=1,
        ksize=(
            3,
            3,
        ),
        mapping_color=[
            46,
            206,
            254,
        ],
        threshold=0.5,
        type='RadialVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
geometry = dict(
    ranges=[
        512,
        896,
        1,
    ], resolution=[
        0.201171875,
        0.2,
    ], size=3)
img_path = 'ADC_Data'
img_suffix = '.npy'
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    auxiliary_head=dict(
        ignore_index=None,
        in_channels=256,
        input_angle_size=224,
        loss_decode=dict(
            classification='FocalLoss',
            loss_weight_classif=2,
            loss_weight_reg=100,
            regression='SmoothL1Loss',
            type='loss_classif_reg'),
        num_classes=1,
        type='Detection_Header'),
    backbone=dict(
        depths=(
            2,
            2,
            15,
            2,
        ),
        dims=[
            64,
            128,
            256,
            512,
        ],
        downsample_version='v3',
        drop_path_rate=0.6,
        forward_type='v05_noz',
        mlp_ratio=4.0,
        norm_layer='ln2d',
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_size=2,
        patchembed_version='v1',
        pretrained='',
        ssm_conv=3,
        ssm_conv_bias=False,
        ssm_d_state=16,
        ssm_dt_rank='auto',
        ssm_ratio=2.0,
        type='FFTRadNet_VSSM_backbone'),
    data_preprocessor=dict(mean=None, std=None, type='RadialDataPreprocessor'),
    decode_head=dict(
        ignore_index=None,
        in_planes=256,
        loss_decode=dict(loss_weight=100, type='loss_seg'),
        num_classes=1,
        type='Segmentation_Header'),
    pretrained=None,
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='RadialEncoderDecoder')
model_wrapper_cfg = dict(
    find_unused_parameters=True, type='MMDistributedDataParallel')
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0001, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0))),
    type='OptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = dict(
    by_epoch=False, gamma=0.9, step_size=16000, type='StepLR')
resume = False
seg_map_path = 'radar_Freespace'
seg_map_suffix = '.png'
statistics = dict(
    input_mean=[],
    input_std=[],
    reg_mean=[
        0.4048094369863972,
        0.3997392847799934,
    ],
    reg_std=[
        0.6968599580482511,
        0.6942950877813826,
    ])
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(
            img_path='ADC_Data/test', seg_map_path='radar_Freespace/test'),
        data_root='segmentation/data/RADIal',
        difficult=True,
        encoder=True,
        geometry=dict(
            ranges=[
                512,
                896,
                1,
            ], resolution=[
                0.201171875,
                0.2,
            ], size=3),
        img_suffix='.npy',
        perform_FFT='ADC',
        pipeline=[
            dict(type='LoadADCDataAndAnnoFromFile'),
            dict(data_mode='ADC', type='PackRadialSegInputs'),
        ],
        seg_map_suffix='.png',
        statistics=dict(
            input_mean=[],
            input_std=[],
            reg_mean=[
                0.4048094369863972,
                0.3997392847799934,
            ],
            reg_std=[
                0.6968599580482511,
                0.6942950877813826,
            ]),
        type='RADIalDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    classification='FocalLoss',
    loss_seg_weight=100,
    loss_weight_classif=2,
    loss_weight_reg=100,
    regression='SmoothL1Loss',
    type='RadialMetric')
test_pipeline = [
    dict(type='LoadADCDataAndAnnoFromFile'),
    dict(data_mode='ADC', type='PackRadialSegInputs'),
]
train_cfg = dict(
    max_iters=160000, type='IterBasedTrainLoop', val_interval=10000)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_prefix=dict(
            img_path='ADC_Data/train', seg_map_path='radar_Freespace/train'),
        data_root='segmentation/data/RADIal',
        difficult=True,
        encoder=True,
        geometry=dict(
            ranges=[
                512,
                896,
                1,
            ], resolution=[
                0.201171875,
                0.2,
            ], size=3),
        img_suffix='.npy',
        perform_FFT='ADC',
        pipeline=[
            dict(type='LoadADCDataAndAnnoFromFile'),
            dict(data_mode='ADC', type='PackRadialSegInputs'),
        ],
        seg_map_suffix='.png',
        statistics=dict(
            input_mean=[],
            input_std=[],
            reg_mean=[
                0.4048094369863972,
                0.3997392847799934,
            ],
            reg_std=[
                0.6968599580482511,
                0.6942950877813826,
            ]),
        type='RADIalDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadADCDataAndAnnoFromFile'),
    dict(data_mode='ADC', type='PackRadialSegInputs'),
]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(type='LoadADCDataAndAnnoFromFile'),
    dict(
        transforms=[
            [
                dict(data_mode='ADC', type='PackRadialSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(
            img_path='ADC_Data/val', seg_map_path='radar_Freespace/val'),
        data_root='segmentation/data/RADIal',
        difficult=True,
        encoder=True,
        geometry=dict(
            ranges=[
                512,
                896,
                1,
            ], resolution=[
                0.201171875,
                0.2,
            ], size=3),
        img_suffix='.npy',
        perform_FFT='ADC',
        pipeline=[
            dict(type='LoadADCDataAndAnnoFromFile'),
            dict(data_mode='ADC', type='PackRadialSegInputs'),
        ],
        seg_map_suffix='.png',
        statistics=dict(
            input_mean=[],
            input_std=[],
            reg_mean=[
                0.4048094369863972,
                0.3997392847799934,
            ],
            reg_std=[
                0.6968599580482511,
                0.6942950877813826,
            ]),
        type='RADIalDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    classification='FocalLoss',
    loss_seg_weight=100,
    loss_weight_classif=2,
    loss_weight_reg=100,
    regression='SmoothL1Loss',
    type='RadialMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/vssm_RADIal_ADC_base'
