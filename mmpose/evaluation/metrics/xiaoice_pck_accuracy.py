# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmpose.registry import METRICS
from ..functional import keypoint_pck_accuracy


@METRICS.register_module()
class XiaoIcePCKAccuracy(BaseMetric):

    def __init__(self,
                 thr: float = 0.05,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.thr = thr if thr is not None else self.dataset_meta.get(
            'pck_thr', 0.05)
        assert isinstance(self.thr, float) or isinstance(self.thr, Sequence), \
            """PCK threshold should be either a float
            number or a sequence of numbers."""
        if isinstance(self.thr, Sequence):
            assert len(self.thr) == self.dataset_meta['num_keypoints'], \
                """Length of PCK threshold should be the
                same as number of joints."""

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.
        Args:
            data_batch (Sequence[dict]): A batch of data
                from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        for data_sample in data_samples:
            # predicted keypoints coordinates, [1, K, D]
            pred_coords = data_sample['pred_instances']['keypoints']
            # ground truth data_info
            gt = data_sample['gt_instances']
            # ground truth keypoints coordinates, [1, K, D]
            gt_coords = gt['keypoints']
            # ground truth keypoints_visible, [1, K, 1]
            mask = gt['keypoints_visible'].astype(bool).reshape(1, -1)

            result = {
                'pred_coords': pred_coords,
                'gt_coords': gt_coords,
                'mask': mask,
            }

            torso_size_ = gt['torso_size']
            torso_size = np.array([torso_size_, torso_size_]).reshape(-1, 2)
            result['torso_size'] = torso_size

            self.results.append(result)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.
        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
            The returned result dict may have the following keys:
                - 'PCK': The pck accuracy normalized by `bbox_size`.
                - 'PCKh': The pck accuracy normalized by `head_size`.
                - 'tPCK': The pck accuracy normalized by `torso_size`.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # pred_coords: [N, K, D]
        pred_coords = np.concatenate(
            [result['pred_coords'] for result in results])
        # gt_coords: [N, K, D]
        gt_coords = np.concatenate([result['gt_coords'] for result in results])
        # mask: [N, K]
        mask = np.concatenate([result['mask'] for result in results])

        metrics = dict()

        norm_size_torso = np.concatenate(
            [result['torso_size'] for result in results])

        logger.info(f'Evaluating {self.__class__.__name__} '
                    f'(normalized by ``"torso_size"``)...')

        _, tpck, _ = keypoint_pck_accuracy(pred_coords, gt_coords, mask,
                                           self.thr, norm_size_torso)
        metrics['tPCK'] = tpck

        return metrics
