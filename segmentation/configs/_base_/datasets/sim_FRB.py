#教程https://mmsegmentation.readthedocs.io/zh-cn/latest/user_guides/1_config.html
# dataset settings
dataset_type = 'simFRBDataset'
data_root = 'data/FAST_FREX/sim_sample'
crop_size = (512, 512)# 训练时的裁剪大小
# #
# custom_imports = dict(
#     imports = ['data.transforms'],
#     allow_failed_imports = False
# )

train_pipeline = [
    # dict(type='LoadFitsFromFile'), # 第1个流程，从文件路径里加载图像
    dict(type='LoadImageFromFile'), # 第1个流程，从文件路径里加载图像
    dict(type='LoadAnnotations', reduce_zero_label=False),
    # dict(type='LoadFitsAnnotations'), # 第2个流程，对于当前图像，加载它的标注图像

    # dict(type='sim_FRB_split'),

    dict(
        type='LabelMapper',# 自定义标签映射
        mapping = {0:0, 255:1},# 将255映射为1
        default_val = -100# 未指定的标签值映射为255
    ),

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
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='LabelMapper',  # 自定义标签映射
        mapping={0: 0, 255: 1},  # 将255映射为1
        default_val=-100  # 未指定的标签值映射为255
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
    persistent_workers=True,# 在一个epoch结束后关闭worker进程，可以加快训练速度
    sampler=dict(type='InfiniteSampler', shuffle=True),# 训练时进行随机洗牌(shuffle)
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

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])# 精度评估方法，我们在这里使用 IoUMetric 进行评估
test_evaluator = val_evaluator