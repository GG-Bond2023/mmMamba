#教程https://mmsegmentation.readthedocs.io/zh-cn/latest/user_guides/1_config.html
# dataset settings

import os

# from classification.data.data_simmim_pt import collate_fn

dataset_type = 'RADIalDataset'
data_root = 'data/RADIal'

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
    "input_mean": [],
    "input_std": [],
    "reg_mean": [0.4048094369863972, 0.3997392847799934],
    "reg_std": [0.6968599580482511, 0.6942950877813826]
}

# crop_size = (512, 512)# 训练时的裁剪大小

train_pipeline = [
    dict(type='LoadADCDataFromFile'),
    # dict(type='LoadImageFromFile'), # 第1个流程，从文件路径里加载图像
    # dict(type='LoadRADIalFromFile'), # 第1个流程，从文件路径里加载RADIal data
    dict(type='LoadAnnotations', reduce_zero_label=True), # 第2个流程，对于当前图像，加载它的标注图像

    # dict(
    #     type='RandomResize',# 调整输入图像大小(resize)和其标注图像的数据增广流程
    #     scale=(2048, 512),# 图像裁剪的大小
    #     ratio_range=(0.5, 2.0),# 数据增广的比例范围
    #     keep_ratio=True),# 调整图像大小时是否保持纵横比
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    # dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadADCDataFromFile'),
    # dict(type='LoadImageFromFile'),
    # dict(type='LoadRADIalFromFile'),  # 第1个流程，从文件路径里加载RADIal data
    # dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]
# img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadADCDataFromFile'),
    # dict(type='LoadImageFromFile', backend_args=None),
    # dict(type='LoadRADIalFromFile'), # 第1个流程，从文件路径里加载RADIal data
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
            [dict(type='LoadAnnotations')],
            [dict(type='PackSegInputs')]
        ]
    )
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,# 在一个epoch结束后关闭worker进程，可以加快训练速度
    sampler=dict(type='InfiniteSampler', shuffle=True),# 训练时进行随机洗牌(shuffle)
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
        perform_FFT='ADC',
        pipeline=train_pipeline
    ),
    collate_fn=dict(type='RADIal_collate')
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
        perform_FFT='ADC',
        data_prefix=dict(img_path=os.path.join( img_path,'val'), seg_map_path=os.path.join(seg_map_path, 'val') ),
        pipeline=test_pipeline
    ),
    collate_fn=dict(type='RADIal_collate')
)
test_dataloader = val_dataloader
test_dataloader = dict(
    batch_size=1,
    data_prefix=dict(img_path=os.path.join(img_path, 'test'), seg_map_path=os.path.join(seg_map_path, 'test')),
)

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])# 精度评估方法，我们在这里使用 IoUMetric 进行评估
test_evaluator = val_evaluator
