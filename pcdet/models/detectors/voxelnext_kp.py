import torch

from .detector3d_template import Detector3DTemplate
from ..model_utils import model_nms_utils


class VoxelNeXt_KP(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):

        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        
        disp_dict = {}
        loss, tb_dict = self.dense_head.get_loss()

        if hasattr(self, 'roi_head') and self.roi_head is not None:
            loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
            loss += loss_rcnn
        
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        if 'batch_kp_preds' in batch_dict:
            post_process_cfg = self.model_cfg.POST_PROCESSING
            batch_size = batch_dict['batch_size']
            recall_dict = {}
            pred_dicts = []
            for index in range(batch_size):
                if batch_dict.get('batch_index', None) is not None:
                    assert batch_dict['batch_box_preds'].shape.__len__() == 2
                    batch_mask = (batch_dict['batch_index'] == index)
                else:
                    assert batch_dict['batch_box_preds'].shape.__len__() == 3
                    batch_mask = index

                box_preds = batch_dict['batch_box_preds'][batch_mask]
                kp_preds = batch_dict['batch_kp_preds'][batch_mask]
                if 'batch_kp_vis_preds' in batch_dict:
                    kp_vis = batch_dict['batch_kp_vis_preds'][batch_mask]
                else:
                    kp_vis = batch_dict.get('roi_kps_vis', None)
                    if kp_vis is not None:
                        kp_vis = kp_vis[batch_mask]
                    else:
                        kp_vis = box_preds.new_ones((kp_preds.shape[0], kp_preds.shape[1]))

                src_box_preds = box_preds

                if not isinstance(batch_dict['batch_cls_preds'], list):
                    cls_preds = batch_dict['batch_cls_preds'][batch_mask]
                    src_cls_preds = cls_preds
                    assert cls_preds.shape[1] in [1, self.num_class]
                    if not batch_dict['cls_preds_normalized']:
                        cls_preds = torch.sigmoid(cls_preds)
                else:
                    cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                    src_cls_preds = cls_preds
                    if not batch_dict['cls_preds_normalized']:
                        cls_preds = [torch.sigmoid(x) for x in cls_preds]

                if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                    if not isinstance(cls_preds, list):
                        cls_preds = [cls_preds]
                        multihead_label_mapping = [torch.arange(1, self.num_class, device=cls_preds[0].device)]
                    else:
                        multihead_label_mapping = batch_dict['multihead_label_mapping']

                    cur_start_idx = 0
                    pred_scores, pred_labels, pred_boxes, pred_kps, pred_kps_vis = [], [], [], [], []
                    for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                        assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                        cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                        cur_kp_preds = kp_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                        cur_kp_vis = kp_vis[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                        cur_pred_scores, cur_pred_labels, cur_pred_boxes, cur_pred_kps = model_nms_utils.multi_classes_nms(
                            cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                            nms_config=post_process_cfg.NMS_CONFIG,
                            score_thresh=post_process_cfg.SCORE_THRESH,
                            kp_preds=cur_kp_preds,
                        )
                        cur_pred_labels = cur_label_mapping[cur_pred_labels]
                        pred_scores.append(cur_pred_scores)
                        pred_labels.append(cur_pred_labels)
                        pred_boxes.append(cur_pred_boxes)
                        pred_kps.append(cur_pred_kps)
                        pred_kps_vis.append(cur_kp_vis[:cur_pred_kps.shape[0]])
                        cur_start_idx += cur_cls_preds.shape[0]

                    final_scores = torch.cat(pred_scores, dim=0)
                    final_labels = torch.cat(pred_labels, dim=0)
                    final_boxes = torch.cat(pred_boxes, dim=0)
                    final_kps = torch.cat(pred_kps, dim=0)
                    final_kps_vis = torch.cat(pred_kps_vis, dim=0)
                else:
                    cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                    if batch_dict.get('has_class_labels', False):
                        label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                        label_preds = batch_dict[label_key][index]
                    else:
                        label_preds = label_preds + 1
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=cls_preds, box_preds=box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH
                    )

                    if post_process_cfg.OUTPUT_RAW_SCORE:
                        max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                        selected_scores = max_cls_preds[selected]

                    final_scores = selected_scores
                    final_labels = label_preds[selected]
                    final_boxes = box_preds[selected]
                    final_kps = kp_preds[selected]
                    final_kps_vis = kp_vis[selected]

                recall_dict = self.generate_recall_record(
                    box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                    recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                    thresh_list=post_process_cfg.RECALL_THRESH_LIST
                )

                record_dict = {
                    'pred_boxes': final_boxes,
                    'pred_kps': final_kps,
                    'pred_kps_vis': final_kps_vis,
                    'pred_scores': final_scores,
                    'pred_labels': final_labels
                }
                pred_dicts.append(record_dict)

            return pred_dicts, recall_dict

        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict
