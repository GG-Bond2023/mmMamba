_base_ = [
    '../_base_/models/upernet_swin.py', '../_base_/datasets/RADIal.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

# data_preprocessor = dict(
#     type='SegDataPreProcessor',
#     mean=None,
#     std=None,
#     # bgr_to_rgb=True,
#     pad_val=0,
#     seg_pad_val=255
# )
data_preprocessor = None

# checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window7_224_20220317-e9b98025.pth'  # noqa
model = dict(
    # data_preprocessor=data_preprocessor,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=48,
        depths=[2, 2, 6, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True
    ),
    decode_head=dict(in_channels=[128, 256, 512, 1024], num_classes=150),
    auxiliary_head=dict(in_channels=512, num_classes=150)
)


# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }
    )
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader

#创建自定义钩子，用来将数据集分成train\val\test
# custom_imports = dict(
#     imports=['segmentation.engine.hooks.Radial_Hook'],
#     allow_failed_imports=False
# )
# custom_hooks = [
#     dict(
#         type = 'RadialSplitHook',
#         data_root = _base_.data_root,
#         img_path = _base_.img_path,
#         seg_map_path = _base_.seg_map_path,
#         img_suffix = _base_.img_suffix,
#         seg_map_suffix = _base_.seg_map_suffix,
#         data_split_info_dir = _base_.data_split_info_dir,
#         data_split_info = _base_.data_split_info,
#         split_ratio = [0.7,0.15,0.15],
#         regenerate_split_info_file = True,#是否重新生成数据集索引文件
#         random = True,#是否为train/val/test随机分配数据集索引文件
#     )
# ]