import os.path as osp
from typing import List, Sequence, Any
from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS
from segmentation.models.losses.Radial_Loss import loss_seg, loss_classif_reg
import torch
@METRICS.register_module()
class RadialMetric(BaseMetric):
    def __init__(self, loss_seg_weight, loss_weight_classif, loss_weight_reg, classification, regression):
        super().__init__()  # self.results is initialized here
        self.loss_seg  = loss_seg(loss_weight=loss_seg_weight)
        self.loss_classif_reg = loss_classif_reg(classification=classification, regression=regression, loss_weight_classif=loss_weight_classif, loss_weight_reg=loss_weight_reg )
        self.output_dir = True
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        # Modify evaluation metrics to use the same loss_seg and loss_classif_reg as training
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the models.
        """

        pred_sem_seg = torch.stack([s['pred_sem_seg']['data'] for s in data_samples])
        gt_sem_seg = torch.stack([s['gt_sem_seg']['data'] for s in data_samples])
        assert pred_sem_seg.ndim ==4 and gt_sem_seg.ndim ==4

        pred_label_map = torch.stack([s['pred_label_map']['data'] for s in data_samples])
        gt_label_map = torch.stack([s['gt_label_map']['data'] for s in data_samples])
        assert pred_label_map.ndim ==4 and gt_label_map.ndim ==4

        loss = {}
        loss[self.loss_seg.loss_name] = self.loss_seg(prediction=pred_sem_seg, target=gt_sem_seg)
        loss[self.loss_classif_reg.loss_name] = self.loss_classif_reg(prediction=pred_label_map, target=gt_label_map)

        self.results.append(loss)

        # for data_sample in data_samples:
        #     loss = {}
        #     pred_sem_seg = data_sample['pred_sem_seg']['data']
        #     gt_sem_seg = data_sample['gt_sem_seg']['data'].to(pred_sem_seg)
        #     assert pred_sem_seg.ndim>=3 and gt_sem_seg.ndim>=3
        #     if pred_sem_seg.ndim==3:    pred_sem_seg_unsqz = pred_sem_seg.unsqueeze(0)
        #     if gt_sem_seg.ndim==3:      gt_sem_seg_unsqz   = gt_sem_seg.unsqueeze(0)
        #     loss[self.loss_seg.loss_name] = self.loss_seg(prediction=pred_sem_seg_unsqz, target=gt_sem_seg_unsqz)
        #
        #     pred_label_map = data_sample['pred_label_map']['data']
        #     gt_label_map = data_sample['gt_label_map']['data'].to(pred_label_map)
        #     assert pred_label_map.ndim>=3 and gt_label_map.ndim>=3
        #     if pred_label_map.ndim==3:    pred_label_map_unsqz = pred_label_map.unsqueeze(0)
        #     if gt_label_map.ndim==3:      gt_label_map_unsqz   = gt_label_map.unsqueeze(0)
        #     loss[self.loss_classif_reg.loss_name] = self.loss_classif_reg(prediction=pred_label_map_unsqz, target=gt_label_map_unsqz)
        #
        #     self.results.append(loss)

        # print('RadialMetric.process:  Length of self.results is:', len(self.results))
        # format_result
            # if self.output_dir is not None:
            #     basename = osp.splitext(osp.basename(
            #         data_sample['img_path']))[0]
            #     png_filename = osp.abspath(
            #         osp.join(self.output_dir, f'{basename}.png'))
            #     output_mask = pred_sem_seg.cpu().numpy()
            #     # The index range of official ADE20k dataset is from 0 to 150.
            #     # But the index range of output is from 0 to 149.
            #     # That is because we set reduce_zero_label=True.
            #     if data_sample.get('reduce_zero_label', False):
            #         output_mask = output_mask + 1
            #     output = Image.fromarray(output_mask.astype(np.uint8))
            #     output.save(png_filename)
    def compute_metrics(self, results: list) -> dict:
        assert 'loss_seg' and 'loss_classif_reg' in results[0]

        # print('RadialMetric.compute_metrics:  Length of self.results is:', len(self.results))

        mean_loss_classif = torch.stack([result['loss_classif_reg'][0] for result in results]).mean()
        mean_loss_reg = torch.stack([result['loss_classif_reg'][1] for result in results]).mean()
        mean_loss_seg = torch.stack([result['loss_seg'] for result in results]).mean()

        val_loss = {'mean_loss_classif': mean_loss_classif,
                    'mean_loss_reg': mean_loss_reg,
                    'mean_loss_seg': mean_loss_seg}
        return val_loss
