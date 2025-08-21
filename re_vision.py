import logging
import os
import io
import argparse
import json
from pathlib import Path
from typing import Union, Sequence, List, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from doclayout_yolo import YOLOv10

yolo_model = None
BBox = Tuple[int, int, int, int, str, float]
BBoxList = List[BBox]


def make_pil_images(
    paths_or_images: List[Union[str, io.BytesIO, Image.Image]],
) -> List[Image.Image]:
    return [
        Image.open(poi).convert("RGB") if isinstance(poi, (str, io.BytesIO)) else poi
        for poi in paths_or_images
    ]


def get_cuda_memory() -> int:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return info.total
    except:
        return 0


def mask_boxes_batch(
    images: List[Image.Image], boxes_list: List[BBoxList]
) -> List[Image.Image]:
    img_arrays = [np.array(img) for img in images]
    masks = [np.zeros(img.shape[:2], dtype=bool) for img in img_arrays]
    for mask, boxes in zip(masks, boxes_list):
        for x1, y1, x2, y2, *_ in boxes:
            mask[int(y1) : int(y2), int(x1) : int(x2)] = True
    masked_arrays = []
    for img_array, mask in zip(img_arrays, masks):
        masked_array = img_array.copy()
        if len(img_array.shape) == 3:
            masked_array[mask] = [255, 255, 255]
        else:
            masked_array[mask] = 255
        masked_arrays.append(masked_array)
    return [Image.fromarray(arr) for arr in masked_arrays]


def _predict_boxes_batch(
    model, images: Sequence[Image.Image], conf: float, device: str
) -> List[BBoxList]:
    try:
        imgsz = getattr(model, "imgsz", 1024)
    except Exception:
        imgsz = 1024
    results = model.predict(images, imgsz=imgsz, conf=conf, device=device)
    out: List[BBoxList] = []
    for res in results:
        boxes: BBoxList = []
        for box in res.boxes:
            xyxy = [int(v) for v in box.xyxy.tolist()[0]]
            cls_idx = int(box.cls.item()) if hasattr(box.cls, "item") else int(box.cls)
            label = model.names[cls_idx]
            conf_score = (
                float(box.conf.item()) if hasattr(box.conf, "item") else float(box.conf)
            )
            boxes.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], label, conf_score))
        out.append(boxes)
    return out


def run_yolo_recursive_batch(
    images: List[Image.Image],
    model_path: str,
    max_runs: int,
    yolo_threshold: float,
    batch_size: int,
) -> List[BBoxList]:
    global yolo_model
    if yolo_model is None:
        from doclayout_yolo.utils import LOGGER

        LOGGER.setLevel(logging.ERROR)
        yolo_model = YOLOv10(os.path.expanduser(model_path))

    device = "cuda" if get_cuda_memory() > 0 else "cpu"
    current_images = images
    all_boxes_batch: List[BBoxList] = [[] for _ in images]

    for _ in range(max_runs):
        new_boxes_batch: List[BBoxList] = []
        for i in range(0, len(current_images), batch_size):
            batch_imgs = current_images[i : i + batch_size]
            boxes_list = _predict_boxes_batch(
                yolo_model, batch_imgs, yolo_threshold, device
            )
            new_boxes_batch.extend(boxes_list)
        if all(len(boxes) == 0 for boxes in new_boxes_batch):
            break
        for i, boxes in enumerate(new_boxes_batch):
            all_boxes_batch[i].extend(boxes)
        current_images = mask_boxes_batch(current_images, new_boxes_batch)

    return all_boxes_batch


def draw_boxes(image: Image.Image, boxes: BBoxList, out_path: str) -> None:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    padding = 6
    for x1, y1, x2, y2, label, conf in boxes:
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        text = f"{label} {conf:.2f}"
        bbox = font.getbbox(text)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        text_bg_top = max(0, y1 - text_h - padding)
        text_bg_bottom = y1
        text_bg_left = x1
        text_bg_right = x1 + text_w + 2 * padding
        draw.rectangle(
            [text_bg_left, text_bg_top, text_bg_right, text_bg_bottom], fill="red"
        )
        draw.text(
            (x1 + padding, text_bg_top + padding // 2), text, fill="white", font=font
        )
    image.save(out_path)


def collect_image_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    return [p for p in path.rglob("*") if p.suffix.lower() in image_exts]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str, help="Path to an image or folder")
    parser.add_argument("model_path", type=str, help="Path to the YOLO model file")
    parser.add_argument("--max-runs", type=int, default=2, help="Max recursive runs")
    parser.add_argument("--save-images", action="store_true", help="Save drawn images")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="YOLO detection confidence threshold (0.0 to 1.0)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for YOLO inference"
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    image_files = collect_image_files(input_path)
    output_dir = Path("yolo_outputs")
    output_dir.mkdir(exist_ok=True)

    jsonl_path = output_dir / "results.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        images = [Image.open(p).convert("RGB") for p in image_files]
        all_boxes_batch = run_yolo_recursive_batch(
            images, args.model_path, args.max_runs, args.threshold, args.batch_size
        )
        for img_path, image, boxes in zip(image_files, images, all_boxes_batch):
            try:
                rel_path = (
                    img_path.relative_to(input_path)
                    if input_path.is_dir()
                    else Path(img_path.name)
                )
                if args.save_images:
                    out_img_path = output_dir / "drawn" / rel_path
                    out_img_path.parent.mkdir(parents=True, exist_ok=True)
                    draw_boxes(image.copy(), boxes, str(out_img_path))

                # Masked image saving removed

                record = {
                    "file": str(rel_path),
                    "boxes": [
                        {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "label": label,
                            "conf": conf,
                        }
                        for (x1, y1, x2, y2, label, conf) in boxes
                    ],
                }
                f.write(json.dumps(record) + "\n")
                print(f"Processed {img_path} → {len(boxes)} boxes")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    print(f"\nAll results written to {jsonl_path}")
