from typing import List
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmseg.registry import MODELS
from torch import Tensor
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample
from mmseg.utils import (OptSampleList, SampleList)
from mmseg.models.utils import resize
from segmentation.data.RADIal_dataset_process import MultiChannelData
@MODELS.register_module()
class RadialEncoderDecoder(EncoderDecoder):
    def __init__(self,**kwargs):
        super(RadialEncoderDecoder, self).__init__(**kwargs)

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Takes effect during the predict phase"""
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""

        x = self.extract_feat(inputs)
        seg_logits = self.decode_head.predict(x, batch_img_metas, self.test_cfg)

        det_logits = self.auxiliary_head.predict(x, batch_img_metas, self.test_cfg)
        # todo: In addition to segmentation_head, we need to add the predict of auxiliary_head
        return dict(seg_logits=seg_logits, det_logits=det_logits)


    def postprocess_result(self,
                           out_logits: Tensor,
                           data_samples: OptSampleList = None) -> SampleList:# Override mmseg/models/segmentors/base.py
        """ Convert results list to `SegDataSample`.
        Args:
            seg_logits (Tensor): The segmentation results, seg_logits from
                models of each input image.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`. Default to None.
        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        assert out_logits['seg_logits'].shape[1:] == data_samples[0].gt_sem_seg.data.shape and out_logits['det_logits'].shape[1:] == data_samples[0].gt_label_map.data.shape

        batch_size, C_seg, H, W = out_logits['seg_logits'].shape
        C_det = out_logits['det_logits'].shape[1]

        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False
        i_out_logits = dict(seg_logits=None,det_logits=None)
        for i in range(batch_size):
            if not only_prediction:
                img_meta = data_samples[i].metainfo
                # remove padding area
                if 'img_padding_size' not in img_meta:
                    padding_size = img_meta.get('padding_size', [0] * 4)
                else:
                    padding_size = img_meta['img_padding_size']
                padding_left, padding_right, padding_top, padding_bottom = padding_size

                # i_seg_logits shape is 1, C, H, W after remove padding
                i_out_logits['seg_logits'] = out_logits['seg_logits'][i:i + 1, :,
                               padding_top:H - padding_bottom,
                               padding_left:W - padding_right]

                # i_seg_logits shape is 1, C, H, W after remove padding
                i_out_logits['det_logits'] = out_logits['det_logits'][i:i + 1, :,
                               padding_top:H - padding_bottom,
                               padding_left:W - padding_right]

                flip = img_meta.get('flip', None)
                if flip:
                    flip_direction = img_meta.get('flip_direction', None)
                    assert flip_direction in ['horizontal', 'vertical']
                    if flip_direction == 'horizontal':
                        i_out_logits['seg_logits'] = i_out_logits['seg_logits'].flip(dims=(3,))
                        i_out_logits['det_logits'] = i_out_logits['det_logits'].flip(dims=(3,))
                    else:
                        i_out_logits['seg_logits'] = i_out_logits['seg_logits'].flip(dims=(2,))
                        i_out_logits['det_logits'] = i_out_logits['det_logits'].flip(dims=(2,))

                i_out_logits['seg_logits'] = i_out_logits['seg_logits'].squeeze(0)
                i_out_logits['det_logits'] = i_out_logits['det_logits'].squeeze(0)
            else:
                i_out_logits['seg_logits'] = out_logits['seg_logits'][i]
                i_out_logits['det_logits'] = out_logits['det_logits'][i]

            data_samples[i].set_data({
                'pred_sem_seg':
                    PixelData(**{'data': i_out_logits['seg_logits']}),
                'pred_label_map':
                    MultiChannelData(**{'data': i_out_logits['det_logits']}),
            })
        return data_samples
