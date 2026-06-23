_base_ = [
    # '../_base_/models/upernet_swin.py',
    '../_base_/datasets/RADIal.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

backbone_norm_cfg = dict(type='LN', requires_grad=True)

# data_preprocessor = dict(
#     type='SegDataPreProcessor',
#     mean=None,
#     std=None,
#     # bgr_to_rgb=True,
#     pad_val=0,
#     seg_pad_val=255
# )
# data_preprocessor = None
data_preprocessor = dict(
    type='RadialDataPreprocessor',
    # mean=None,
    # std=None,
)

# checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window7_224_20220317-e9b98025.pth'  # noqa
model = dict(
    # _delete_ = True,
    type='RadialEncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,  # Load pretrained backbone
    backbone=dict(
        type='FFTRadNet_ViT_backbone',  # Backbone type, see mmseg/models/backbones/ for details
        # pretrain_img_size=None,
        patch_size = 2,
        in_chans = 32,
        embed_dim = 48,
        strides = [2,2,2,2],
        depths = [2,2,6,2],
        num_heads = [3,6,12,24],
        drop_rates = [0.0, 0.0, 0.0],
        norm_cfg = backbone_norm_cfg
    ),
    decode_head=
        dict(
            type='Segmentation_Header',  # Decode head type. See mmseg/models/decode_heads for available options
            in_planes= 256,
            num_classes= 1,
            ignore_index = None,
            loss_decode=
                dict(type='loss_seg',
                    loss_weight=100
                ),
        ),
    auxiliary_head=dict(
        type = 'Detection_Header',
        in_channels=256,
        num_classes=1,
        input_angle_size=56 * 4,
        ignore_index=None,
        loss_decode=
            dict(type='loss_classif_reg',
                classification = 'FocalLoss',
                regression = 'SmoothL1Loss',
                loss_weight_classif=2,
                loss_weight_reg=100,
            )
    ),
    # models training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999))
)
optimizer = dict(_delete_=True)

param_scheduler = dict(
    _delete_=True,
    type='StepLR',
    by_epoch=False,  # Adjust learning rate by iteration rather than epoch
    step_size=17540,  # Decay every 1000 iterations (adjust based on total iterations)
    gamma=0.9
)

# New: configure DDP parameters
model_wrapper_cfg=dict(
    type='MMDistributedDataParallel',  # MMSeg default DDP wrapper class
    find_unused_parameters=True  # Allow unused parameters in model forward (backbone.vit.norm0), fix DDP error
)


# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=4)
test_dataloader = dict(batch_size=4)

# Configure evaluator loss function
val_evaluator = dict(
    _delete_=True,
    metrics = [
        dict(
            type = 'RadialMetric',
            loss_seg_weight = model['decode_head']['loss_decode']['loss_weight'],
            loss_weight_classif = model['auxiliary_head']['loss_decode']['loss_weight_classif'],
            loss_weight_reg=model['auxiliary_head']['loss_decode']['loss_weight_reg'],
            classification = model['auxiliary_head']['loss_decode']['classification'],
            regression = model['auxiliary_head']['loss_decode']['regression'],
        ),
        dict(
            type='SegDetIoUMetrics',
            geometry=_base_['geometry'],
            statistics=_base_['statistics']
        )
    ]
)
test_evaluator = val_evaluator

train_cfg = dict(type='IterBasedTrainLoop', max_iters=175400, val_interval=1754)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # visualization=dict(
    #     type='RadialVisualizationHook',
    #     interval=1,
    #     draw=True,
    #     threshold=0.5,
    #     mapping_color=[46,206,254],
    #     ksize=(3,3),
    #     dilated_iter=5,
    # ),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=17540,
    ),
    logger=dict(
        type='LoggerHook',
        interval=100,
        log_metric_by_epoch=False
    ),
)
