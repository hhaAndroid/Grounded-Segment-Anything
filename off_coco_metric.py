from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

gt_path = 'dataset/coco10/annotations/instances_val2017.json'
pred_path = 'outputs/annotations/instances_val2017_pred.json'
cocoGt = COCO(gt_path)
cocoDt = COCO(pred_path)

for metric in ['bbox', 'segm']:
    coco_eval = COCOeval(cocoGt, cocoDt, iouType=metric)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
