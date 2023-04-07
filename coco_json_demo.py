import argparse
import os
import copy

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import build_sam, SamPredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
import json
import pycocotools.mask as mask_util


def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, cpu_only=False):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def traverse_files(folder_path):
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    return all_files


def encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code.

    Args:
        mask_results (list): bitmap mask results.

    Returns:
        list | tuple: RLE encoded mask.
    """
    encoded_mask_results = []
    for mask in mask_results:
        encoded_mask_results.append(
            mask_util.encode(
                np.array(mask[:, :, np.newaxis], order='F',
                         dtype='uint8'))[0])  # encoded with RLE
    return encoded_mask_results


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--config_file", "-c", type=str,
                        default='GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
                        help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, default='groundingdino_swint_ogc.pth', help="path to checkpoint file"
    )
    parser.add_argument("--json", type=str,
                        default='/home/PJLAB/huanghaian/dataset/coco1/annotations/instances_val2017.json',
                        help="path to image file")
    parser.add_argument("--image-dir", "-i", type=str, default='/home/PJLAB/huanghaian/dataset/coco1/val2017',  help="path to image file")
    parser.add_argument("--text_prompt", "-t", type=str, default='coco_cls_name.txt', help="text prompt")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    text_prompt = args.text_prompt
    with open(text_prompt, 'r') as f:
        coco_cls_str = f.read()
    text_prompt = coco_cls_str.replace('\n', ',')

    cls_name = coco_cls_str.split('\n')
    cls_name1 = [cls.split(' ') for cls in cls_name]

    ann_json = args.json
    coco = COCO(ann_json)

    json_dir_name = os.path.dirname(ann_json)
    output_json_name = os.path.basename(ann_json)[:-5] + '_pred.json'
    output_name = os.path.join(json_dir_name, output_json_name)

    json_data = coco.dataset
    new_json_data = {
        'info': json_data['info'],
        'licenses': json_data['licenses'],
        'categories': json_data['categories'],
        'images': [],
        'annotations': []
    }
    name2id = {}
    for categories in new_json_data['categories']:
        name2id[categories['name']] = categories['id']

    box_threshold = args.box_threshold
    text_threshold = args.box_threshold

    # load model
    model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)
    predictor = SamPredictor(build_sam(checkpoint="./sam_vit_h_4b8939.pth"))

    img_ids = coco.getImgIds()

    for image_id in img_ids:
        raw_img_info = coco.loadImgs([image_id])[0]
        new_json_data['images'].append(raw_img_info)

        file_name = raw_img_info['file_name']
        image_path = os.path.join(args.image_dir, file_name)
        print(image_path)
        # load image
        image_pil, image = load_image(image_path)

        # run model
        boxes_filt, pred_phrases = get_grounding_output(
            model, image, text_prompt, box_threshold, text_threshold, cpu_only=args.cpu_only
        )
        normalized_boxes = copy.deepcopy(boxes_filt)

        # visualize pred
        size = image_pil.size
        pred_dict = {
            "boxes": normalized_boxes,
            "size": [size[1], size[0]],  # H,W
            "labels": pred_phrases,
        }
        # print(pred_dict['labels'])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        H, W = size[1], size[0]

        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        pred_dict['boxes'] = boxes_filt

        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2])

        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False
        )
        pred_dict['masks'] = masks.numpy()
        pred_dict['boxes'] = pred_dict['boxes'].int().numpy().tolist()

        annotations = []

        for i in range(len(pred_dict['boxes'])):
            label = pred_dict['labels'][i][:-6]

            for cls in cls_name1:
                if label in cls:
                    cls_nam = ' '.join(cls)
                    break

            bbox = pred_dict['boxes'][i]
            coco_bbox = [
                bbox[0],
                bbox[1],
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
            ]

            annotation = dict(
                id=len(annotations) + 1,  # coco api requires id starts with 1
                image_id=image_id,
                bbox=coco_bbox,
                iscrowd=0,
                category_id=name2id[cls_nam],
                area=coco_bbox[2] * coco_bbox[3])

            mask = pred_dict['masks'][i]
            encode_masks = encode_mask_results(mask)
            for encode_mask in encode_masks:
                if isinstance(encode_mask, dict) and isinstance(
                        encode_mask['counts'], bytes):
                    encode_mask['counts'] = encode_mask['counts'].decode()
            annotation['segmentation'] = encode_mask
            annotations.append(annotation)
        new_json_data['annotations'] = annotations

    with open(output_name, "w") as f:
        json.dump(new_json_data, f)

