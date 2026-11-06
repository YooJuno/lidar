"""Center-origin Studio boxes, converted explicitly to MMDetection3D boxes."""
import numpy as np
from mmdet3d.datasets import Det3DDataset
from mmdet3d.registry import DATASETS
from mmdet3d.registry import METRICS
from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.evaluation import IndoorMetric
from mmengine.logging import MMLogger
import torch


@DATASETS.register_module()
class StudioDataset(Det3DDataset):
    METAINFO={'classes':()}

    def parse_ann_info(self, info):
        annotations=super().parse_ann_info(info)
        if annotations is None:annotations=dict(gt_bboxes_3d=np.empty((0,9),dtype=np.float32),gt_labels_3d=np.empty(0,dtype=np.int64))
        annotations['gt_bboxes_3d']=LiDARInstance3DBoxes(annotations['gt_bboxes_3d'],box_dim=9,origin=(.5,.5,.5)).convert_to(self.box_mode_3d)
        return annotations


@METRICS.register_module()
class StudioBoxMetric(IndoorMetric):
    """Evaluate box geometry only; the training head also has velocity slots."""

    def process(self, data_batch, data_samples):
        start=len(self.results)
        super().process(data_batch,data_samples)
        for index in range(start,len(self.results)):
            annotation,prediction=self.results[index]
            annotation=dict(annotation);prediction=dict(prediction)
            for values,key in ((annotation,'gt_bboxes_3d'),(prediction,'bboxes_3d')):
                boxes=values[key]
                values[key]=type(boxes)(boxes.tensor[:,:7],box_dim=7)
            self.results[index]=(annotation,prediction)

    def compute_metrics(self, results):
        # AP has no definition for a class with no validation ground truth.
        # Report the classes actually measured instead of averaging NaNs or
        # inventing perfect scores for missing categories.
        classes=self.dataset_meta['classes']
        present=sorted({int(label) for annotation,_ in results for label in annotation['gt_labels_3d']})
        if not present:raise ValueError('Box AP requires annotated validation objects')
        if len(present)==len(classes):return super().compute_metrics(results)
        missing=[name for index,name in enumerate(classes)if index not in present]
        MMLogger.get_current_instance().warning('No validation ground truth; AP unavailable for: %s',', '.join(missing))
        measured=[]
        for annotation,prediction in results:
            annotation=dict(annotation);prediction=dict(prediction)
            annotation['gt_labels_3d']=np.array([present.index(int(label))for label in annotation['gt_labels_3d']],dtype=np.int64)
            labels=prediction['labels_3d'];mask=torch.zeros_like(labels,dtype=torch.bool)
            for label in present:mask|=labels==label
            for key in ('labels_3d','scores_3d','bboxes_3d'):prediction[key]=prediction[key][mask]
            prediction['labels_3d']=torch.tensor([present.index(int(label))for label in prediction['labels_3d']],dtype=torch.long)
            measured.append((annotation,prediction))
        original=self.dataset_meta
        try:
            self.dataset_meta=dict(original,classes=[classes[i]for i in present])
            return super().compute_metrics(measured)
        finally:self.dataset_meta=original
