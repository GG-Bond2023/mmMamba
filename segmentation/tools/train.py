# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
import mmengine
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
# from mmengine.runner import Runner
from segmentation.engine.runner.Runner import myRunner

from mmseg.registry import RUNNERS

from segmentation.models.backbones import *
from segmentation.models.decode_heads import radial_decode_head
from segmentation.models.losses import *
from segmentation.data import RADIal_dataset_process
from segmentation.models.segmentors.Radial_encoder_decoder import *
from segmentation.evaluation.metrics.radial_metric import *
from segmentation.evaluation.metrics.SegDetIoUMetrics import SegDetIoUMetrics
from segmentation.engine.hooks.visualization_hook import RadialVisualizationHook

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume training
    cfg.resume = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = myRunner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # runner._work_dir = os.path.join(runner._work_dir, runner._timestamp)# Modify storage destination
    # mmengine.mkdir_or_exist(runner.work_dir)


    # if hasattr(runner.model, 'module'):
    #     Module = runner.model.module
    #     # Custom module: backbone.vit.norm0 is not used in RA_decoder, which will cause an error in distributed training and needs to be handled as follows.
    #     for name, param in Module.named_parameters():
    #         if "backbone.vit.norm0" in name:
    #             param.requires_grad = False
    #             print(f"Parameter {name} has been frozen")

    if cfg["model"]["backbone"]["type"]=="FFTRadNet_VSSM_backbone" and cfg['data_mode']=='ADC' and cfg['model']["backbone"]['SelectiveScanMode']=='3D':
        # If selectiveScanMode is 3D and the input data is RADIal-ADC, since vmamba uses lazyInit to initialize weights, manually feed a dummy input after model construction and before training to trigger initialization
        dummy_input = torch.complex(torch.randn(4, 16, 512, 256), torch.randn(4, 16, 512, 256)).to("cuda:0")
        model = runner.model
        model.eval()
        with torch.no_grad():
            model(dummy_input)


    # start training
    runner.train()


if __name__ == '__main__':
    main()
