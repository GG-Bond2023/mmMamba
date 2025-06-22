# --------------------------------------------------------
# Modified by $@#Anonymous#@$
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import json
import random
import argparse
import datetime
import tqdm
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy # 导入损失函数
from timm.utils import accuracy, AverageMeter# 导入准确率计算和指标平均工具

from config import get_config # 导入配置解析函数
from models import build_model # 导入模型构建函数
from data import build_loader  # 导入数据加载器构建函数
from utils.lr_scheduler import build_scheduler  # 导入学习率调度器构建函数
from utils.optimizer import build_optimizer  # 导入优化器构建函数
from utils.logger import create_logger  # 导入日志创建函数
from utils.utils import  NativeScalerWithGradNormCount, auto_resume_helper, reduce_tensor  # 导入混合精度相关工具、自动恢复工具、张量归约工具
from utils.utils import load_checkpoint_ema, load_pretrained_ema, save_checkpoint_ema  # 导入EMA相关的加载/保存函数


from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count  # 导入计算FLOPs的工具

from timm.utils import ModelEma as ModelEma  # 导入模型指数移动平均（EMA）类

# 设置多进程启动方式为'spawn'（避免某些环境下的兼容性问题）
if torch.multiprocessing.get_start_method() != "spawn":
    print(f"||{torch.multiprocessing.get_start_method()}||", end="")
    torch.multiprocessing.set_start_method("spawn", force=True)

def str2bool(v):
    """
    Converts string to bool type; enables command line
    arguments in the format of '--arg1 true --arg2 false'
    将字符串转换为布尔类型，支持命令行参数如'--arg true/false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, metavar="FILE", default="", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, default="/dataset/ImageNet_ILSVRC2012", help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', default=time.strftime("%Y%m%d%H%M%S", time.localtime()), help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')#控制是否使用融合的层归一化（Fused Layer Normalization）。融合层归一化是一种优化的层归一化实现方式，它将多个操作合并为一个核函数，能减少GPU内存访问次数，从而提升计算效率。
    parser.add_argument('--optim', type=str, help='overwrite optimizer if provided, can be adamw/sgd.')

    # EMA related parameters    # EMA相关参数
    parser.add_argument('--model_ema', type=str2bool, default=True)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')

    parser.add_argument('--memory_limit_rate', type=float, default=-1, help='limitation of gpu memory use')

    args, unparsed = parser.parse_known_args()

    config = get_config(args) # 解析配置文件和命令行参数

    return args, config


def main(config, args):
    # 构建数据加载器（包括训练集、验证集及其加载器，混合精度增强函数）  mixup_fn：通过线性插值混合两个样本及其标签，或将一个图像的部分区域裁剪并粘贴到另一个图像上，同时调整标签。
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)# 根据配置构建模型

    if dist.get_rank() == 0:# 仅主进程打印模型信息
        if hasattr(model, 'flops'):#检查模型是否具有计算浮点运算次数（FLOPs）的属性或方法
            logger.info(str(model))
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"number of params: {n_parameters}")
            flops = model.flops()# 计算FLOPs
            logger.info(f"number of GFLOPs: {flops / 1e9}")
        else:            # 使用fvcore工具计算FLOPs
            logger.info(flop_count_str(FlopCountAnalysis(model, (dataset_val[0][0][None],))))
    torch.cuda.empty_cache() # 清理GPU缓存
    dist.barrier() # 分布式同步点
    model.cuda()
    model_without_ddp = model# 保存未包装的模型（用于加载检查点）

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        # 创建EMA模型（需在CUDA、DP包装、AMP之后，同步BN和DDP之前）
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',# 可选EMA模型存放设备
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)


    optimizer = build_optimizer(config, model, logger)
    # 包装为分布式数据并行模型
    model = torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False)
    # 创建混合精度损失缩放器（带梯度范数统计）
    loss_scaler = NativeScalerWithGradNormCount()

    # 根据梯度累积步数调整学习率调度器的更新频率
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS) # 每个周期的更新步数
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    # 根据数据增强和标签平滑配置选择损失函数
    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        # 混合精度增强时使用软目标交叉熵
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        # 标签平滑时使用标签平滑交叉熵
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:        # 普通情况使用标准交叉熵
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0# 记录最佳准确率（非EMA模型）
    max_accuracy_ema = 0.0# 记录最佳准确率（EMA模型）

    if config.TRAIN.AUTO_RESUME:
        # 自动恢复训练：查找最新检查点
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file# 更新恢复路径
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        # 加载检查点（包括模型、优化器、调度器、EMA模型等）
        max_accuracy, max_accuracy_ema = load_checkpoint_ema(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger, model_ema)
        # 验证当前模型性能
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if model_ema is not None:
            # 验证EMA模型性能
            acc1_ema, acc5_ema, loss_ema = validate(config, data_loader_val, model_ema.ema)
            logger.info(f"Accuracy of the network ema on the {len(dataset_val)} test images: {acc1_ema:.1f}%")

        if config.EVAL_MODE:# 仅评估模式下提前退出
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        # 加载预训练权重（非恢复训练场景）
        load_pretrained_ema(config, model_without_ddp, logger, model_ema)
        # 验证预训练模型性能
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if model_ema is not None:
            acc1_ema, acc5_ema, loss_ema = validate(config, data_loader_val, model_ema.ema)
            logger.info(f"Accuracy of the network ema on the {len(dataset_val)} test images: {acc1_ema:.1f}%")
        
        if config.EVAL_MODE:
            return

    if config.THROUGHPUT_MODE and (dist.get_rank() == 0):
        logger.info(f"throughput mode ==============================")
        # 测试模型推理吞吐量
        throughput(data_loader_val, model, logger)
        if model_ema is not None:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # 测试EMA模型推理吞吐量
            throughput(data_loader_val, model_ema.ema, logger)
        return


    logger.info("Start training")
    start_time = time.time()
    # 主训练循环
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        # 设置训练数据采样器的 epoch（用于分布式训练的洗牌）
        data_loader_train.sampler.set_epoch(epoch)
        # 执行单轮训练
        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, model_ema)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            # 保存检查点（主进程执行）
            save_checkpoint_ema(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger, model_ema, max_accuracy_ema)

        # 验证当前模型
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')
        if model_ema is not None:
            # 验证EMA模型
            acc1_ema, acc5_ema, loss_ema = validate(config, data_loader_val, model_ema.ema)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1_ema:.1f}%")
            max_accuracy_ema = max(max_accuracy_ema, acc1_ema)
            logger.info(f'Max accuracy ema: {max_accuracy_ema:.2f}%')


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, model_ema=None, model_time_warmup=50):
    # 设置模型为训练模式
    model.train()
    # 清空梯度
    optimizer.zero_grad()

    num_steps = len(data_loader)# 总步数（批次数量）
    # 初始化指标记录器
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        # 清理GPU峰值内存统计（用于监控内存使用）
        torch.cuda.reset_peak_memory_stats()
        # 将数据移动到GPU（非阻塞模式）
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            # 应用混合精度增强（数据和标签同时增强）
            samples, targets = mixup_fn(samples, targets)

        data_time.update(time.time() - end)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)
        loss = criterion(outputs, targets)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        # this attribute is added by timm on one optimizer (adahessian)
        # 判断是否为二阶优化器（如AdaHessian）
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        # 混合精度反向传播：计算梯度、缩放损失、更新参数
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,# 梯度裁剪
                                parameters=model.parameters(), create_graph=is_second_order,# 是否创建二阶导数图
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0) # 是否在当前步更新梯度
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad() # 清空梯度（梯度累积时仅在最后一步清空）
            # 更新学习率调度器（根据累积步数计算当前进度）
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
            if model_ema is not None:
                model_ema.update(model)# 更新EMA模型参数
        # 获取当前损失缩放值（用于监控混合精度状态）
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize() # 等待GPU操作完成
        # 更新指标记录器
        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update# 非累积步时grad_norm为None
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        # 跳过预热阶段的批次（前model_time_warmup个批次），避免初始化开销影响统计
        if idx > model_time_warmup:
            model_time.update(batch_time.val - data_time.val)

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            # 计算当前GPU最大内存占用（MB）
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            # 估算剩余训练时间
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'data time {data_time.val:.4f} ({data_time.avg:.4f})\t'
                f'model time {model_time.val:.4f} ({model_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()# 设置模型为评估模式（关闭Dropout等训练特定层）

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter() # 记录top-1准确率
    acc5_meter = AverageMeter() # 记录top-5准确率

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        # 将数据移至GPU
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output        # 前向传播（使用混合精度）
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))# 计算top-1和top-5准确率
        # 在分布式训练中同步各进程的指标
        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)
        # 更新指标记录器
        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]# 获取批次大小
        for i in range(50):        # 预热阶段（执行50次推理，让GPU达到稳定状态）
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    args, config = parse_option()
    # 检测并提示Apex AMP已弃用（建议使用PyTorch原生AMP）
    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")
    # 初始化分布式训练环境
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])# 获取当前进程的全局排名
        world_size = int(os.environ['WORLD_SIZE'])# 获取总进程数
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    # 设置当前设备并初始化分布式进程组
    torch.cuda.set_device(rank)# 使用NCCL后端（适用于GPU分布式训练） # 通过环境变量自动发现其他进程
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    dist.barrier() # 同步所有进程，确保初始化完成
    # 设置随机种子（确保实验可复现性）
    seed = config.SEED + dist.get_rank() # 每个进程使用不同的种子（基于全局排名）
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True    # 配置cuDNN以获得最佳性能 # 启用cuDNN自动调优，为网络寻找最佳卷积算法

    if True: 
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True # 确保结果可复现（可能影响性能）

    # 根据总批次大小线性缩放学习率（常见的超参数调整策略）
    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    # 如果使用梯度累积，还需要进一步缩放学习率
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    # 更新配置中的学习率参数（需要临时解冻配置）
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    # 确保所有进程使用相同的输出路径（通过广播主进程的配置）
    # to make sure all the config.OUTPUT are the same
    config.defrost()
    if dist.get_rank() == 0:
        obj = [config.OUTPUT]# 主进程提供输出路径
        # obj = [str(random.randint(0, 100))] # for test
    else:
        obj = [None] # 其他进程接收路径
    dist.broadcast_object_list(obj)# 广播对象列表
    dist.barrier()
    config.OUTPUT = obj[0] # 更新所有进程的输出路径
    print(config.OUTPUT, flush=True)
    config.freeze()
    # 创建输出目录并初始化日志记录器
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    # 主进程保存完整配置到JSON文件
    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # 记录配置信息和命令行参数
    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    # 限制GPU内存使用（可选）
    if args.memory_limit_rate > 0 and args.memory_limit_rate < 1:
        torch.cuda.set_per_process_memory_fraction(args.memory_limit_rate)
        usable_memory = torch.cuda.get_device_properties(0).total_memory * args.memory_limit_rate / 1e6
        print(f"===========> GPU memory is limited to {usable_memory}MB", flush=True)

    main(config, args)
