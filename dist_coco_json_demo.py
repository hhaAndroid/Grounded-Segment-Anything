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


def load_model(model_config_path, model_checkpoint_path, cpu_only=True):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, off_load=False):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."

    if off_load:
        model = model.to(get_device())
        image = image.to(get_device())
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    if off_load:
        model = model.to('cpu')

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


from torch.utils.data import DataLoader, Dataset
from mmengine.dataset import DefaultSampler, default_collate, worker_init_fn
from functools import partial
from mmengine.dist import (broadcast, get_dist_info, get_rank, init_dist,
                           is_distributed, master_only, collect_results,barrier)
from mmengine.device import get_device
from torch.nn.parallel import DistributedDataParallel


class SimpleDataset(Dataset):
    def __init__(self, img_ids):
        self.img_ids = img_ids

    def __getitem__(self, item):
        return self.img_ids[item]

    def __len__(self):
        return len(self.img_ids)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--config_file", "-c", type=str,
                        default='GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
                        help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, default='groundingdino_swint_ogc.pth', help="path to checkpoint file"
    )
    parser.add_argument("--data-root", type=str, default='/home/PJLAB/huanghaian/dataset/coco10/')
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", help="output directory"
    )
    parser.add_argument("--ann-file", type=str, default='annotations/instances_val2017.json')
    parser.add_argument("--data-prefix", type=str, default='val2017/')
    parser.add_argument("--text_prompt", "-t", type=str, default='coco_cls_name.txt', help="text prompt")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--off_load", action="store_false")
    parser.add_argument("--sam_cpu_only", action="store_false")
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    off_load = args.off_load
    sam_cpu_only = args.sam_cpu_only
    print('off_load:', off_load, 'sam_cpu_only:', sam_cpu_only)

    if args.launcher == 'none':
        _distributed = False
    else:
        _distributed = True

    if _distributed and not is_distributed():
        init_dist(args.launcher)

    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    text_prompt = args.text_prompt
    with open(text_prompt, 'r') as f:
        coco_cls_str = f.read()
    text_prompt = coco_cls_str.replace('\n', ',')

    cls_name = coco_cls_str.split('\n')
    cls_name1 = [cls.split(' ') for cls in cls_name]

    coco = COCO(os.path.join(args.data_root, args.ann_file))

    name2id = {}
    for categories in coco.dataset['categories']:
        name2id[categories['name']] = categories['id']

    coco_dataset = SimpleDataset(coco.getImgIds())

    print('data_len', len(coco_dataset), 'num_word_size', get_dist_info()[1])

    sampler = DefaultSampler(coco_dataset, False)
    num_workers = 0
    init_fn = partial(
        worker_init_fn,
        num_workers=num_workers,
        rank=get_rank(),
        seed=0,
        disable_subprocess_warning=True)
    data_loader = DataLoader(
        dataset=coco_dataset,
        sampler=sampler,
        collate_fn=lambda x: x,
        worker_init_fn=init_fn,
        batch_size=1,
        num_workers=num_workers,
        persistent_workers=False,
        drop_last=False)

    box_threshold = args.box_threshold
    text_threshold = args.box_threshold

    # load model
    model = load_model(config_file, checkpoint_path)
    predictor = SamPredictor(build_sam(checkpoint="./sam_vit_h_4b8939.pth"))

    if _distributed:
        model = model.to(get_device())
        model = DistributedDataParallel(
            module=model,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False,
            find_unused_parameters=False)
        predictor.model = predictor.model.to(get_device())
        predictor.model = DistributedDataParallel(
            module=predictor.model,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False,
            find_unused_parameters=False)
    else:
        if not off_load:
            model = model.to(get_device())
            if not sam_cpu_only:
                predictor.model = predictor.model.to(get_device())

    annotations = []
    coco_preds = {}

    part_json_data = []

    for i, data in enumerate(data_loader):
        new_json_data = dict(annotation=[])
        image_id = data[0]
        raw_img_info = coco.loadImgs([image_id])[0]
        new_json_data['image'] = raw_img_info

        file_name = raw_img_info['file_name']
        image_path = os.path.join(args.data_root, args.data_prefix, file_name)

        if get_rank() == 0:
            print('len:', len(data_loader), 'iter', i+1)

        # load image
        image_pil, image = load_image(image_path)

        # run model
        boxes_filt, pred_phrases = get_grounding_output(
            model, image, text_prompt, box_threshold, text_threshold, off_load=off_load)
        if boxes_filt.shape[0] == 0:
            part_json_data.append(new_json_data)
            continue

        normalized_boxes = copy.deepcopy(boxes_filt)

        # visualize pred
        size = image_pil.size
        pred_dict = {
            "boxes": normalized_boxes,
            "size": [size[1], size[0]],  # H,W
            "labels": pred_phrases,
        }

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if off_load and not sam_cpu_only:
            predictor.model = predictor.model.to(get_device())

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

        if off_load and not sam_cpu_only:
            predictor.model = predictor.model.to('cpu')

        pred_dict['masks'] = masks.numpy()
        pred_dict['boxes'] = pred_dict['boxes'].int().numpy().tolist()

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
            new_json_data['annotation'].append(annotation)

        part_json_data.append(new_json_data)

    all_json_results = collect_results(part_json_data, len(coco_dataset), 'cpu')

    if get_rank() == 0:
        new_json_data = {'info': coco.dataset['info'], 'licenses': coco.dataset['licenses'],
                         'categories': coco.dataset['categories'],
                         'images': [json_results['image'] for json_results in all_json_results]}

        annotations = []
        annotation_id = 1
        for annotation in all_json_results:
            annotation = annotation['annotation']
            for ann in annotation:
                ann['id'] = annotation_id
                annotation_id += 1
                annotations.append(ann)

        if len(annotations) > 0:
            new_json_data['annotations'] = annotations

        output_json_name = args.ann_file[:-5] + '_pred.json'
        output_name = os.path.join(args.output_dir, output_json_name)
        os.makedirs(os.path.dirname(output_name), exist_ok=True)

        with open(output_name, "w") as f:
            json.dump(new_json_data, f)
