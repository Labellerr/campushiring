# inference_and_eval.py
import json, os
from ultralytics import YOLO
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def run_inference_and_save(model_path, dataset_json, images_dir, out_json_path, conf=0.25, imgsz=640):
    model = YOLO(model_path)
    coco = COCO(dataset_json)
    img_ids = [img['id'] for img in coco.loadImgs(coco.getImgIds())]
    predictions = []
    from pycocotools import mask as maskUtils
    import cv2
    for img in coco.loadImgs(img_ids):
        img_path = os.path.join(images_dir, img['file_name'])
        res = model.predict(img_path, imgsz=imgsz, conf=conf)
        r = res[0]
        if hasattr(r, "masks") and r.masks is not None:
            masks = r.masks.data.cpu().numpy()
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            for i in range(len(masks)):
                mask = masks[i].astype('uint8')
                rle = maskUtils.encode(np.asfortranarray(mask))
                rle['counts'] = rle['counts'].decode('ascii')
                area = int(mask.sum())
                x1,y1,x2,y2 = boxes[i].tolist()
                bbox = [float(x1), float(y1), float(x2-x1), float(y2-y1)]
                predictions.append({
                    "image_id": img['id'],
                    "category_id": int(classes[i]),
                    "segmentation": rle,
                    "score": float(scores[i]),
                    "bbox": bbox,
                    "area": area
                })

    json.dump(predictions, open(out_json_path, "w"))
    print("Wrote predictions to", out_json_path)
    return out_json_path

def evaluate_coco(gt_json, pred_json):
    cocoGt = COCO(gt_json)
    cocoDt = cocoGt.loadRes(pred_json)
    cocoEval = COCOeval(cocoGt, cocoDt, iouType='segm')
    cocoEval.params.useSegm = True
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()  # prints standard COCO metrics
    return cocoEval.stats

if __name__ == "__main__":
    model_path = "runs/seg/yolov8n_seg_labellerr/weights/last.pt"
    gt = "dataset/annotations/instances_test.json"
    images_dir = "dataset/images"
    out_pred = "predictions/test_predictions_coco.json"
    run_inference_and_save(model_path, gt, images_dir, out_pred)
    stats = evaluate_coco(gt, out_pred)
    print("COCO stats (APs):", stats)
