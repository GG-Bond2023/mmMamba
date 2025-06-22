#教程https://mmsegmentation.readthedocs.io/zh-cn/latest/user_guides/1_config.html
# dataset settings
dataset_type = 'ADE20KDataset'
data_root = 'data/ade/ADEChallengeData2016'
crop_size = (512, 512)# 训练时的裁剪大小
train_pipeline = [
    dict(type='LoadImageFromFile'), # 第1个流程，从文件路径里加载图像
    dict(type='LoadAnnotations', reduce_zero_label=True), # 第2个流程，对于当前图像，加载它的标注图像
    dict(
        type='RandomResize',# 调整输入图像大小(resize)和其标注图像的数据增广流程
        scale=(2048, 512),# 图像裁剪的大小
        ratio_range=(0.5, 2.0),# 数据增广的比例范围
        keep_ratio=True),# 调整图像大小时是否保持纵横比
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
    dict(type='LoadAnnotations', reduce_zero_label=True),
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
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,# 在一个epoch结束后关闭worker进程，可以加快训练速度
    sampler=dict(type='InfiniteSampler', shuffle=True),# 训练时进行随机洗牌(shuffle)
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/training', seg_map_path='annotations/training'),
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
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])# 精度评估方法，我们在这里使用 IoUMetric 进行评估
test_evaluator = val_evaluator
