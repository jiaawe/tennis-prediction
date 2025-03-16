import argparse
import os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import glob
import json

# please make sure https://github.com/IDEA-Research/GroundingDINO is installed correctly.
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

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
    if "model" in checkpoint:
        checkpoint = checkpoint["model"]
    load_res = model.load_state_dict(clean_state_dict(checkpoint), strict=False)
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
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    logits_filt = logits.cpu().clone()
    boxes_filt = boxes.cpu().clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    confidence_scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        confidence_scores.append(float(logit.max().item()))
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases, confidence_scores

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

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask

def convert_boxes_to_json(boxes, confidence_scores, image_size):
    """Convert boxes to JSON format with absolute coordinates."""
    W, H = image_size
    json_boxes = []
    
    for box, conf in zip(boxes, confidence_scores):
        # Convert from relative [cx, cy, w, h] to absolute [x1, y1, x2, y2]
        box = box * torch.Tensor([W, H, W, H])
        cx, cy, w, h = box.tolist()
        x1 = int(cx - w/2)
        y1 = int(cy - h/2)
        x2 = int(cx + w/2)
        y2 = int(cy + h/2)
        
        json_boxes.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": conf
        })
    
    return json_boxes

def process_images_in_folder(args):
    # Get video name from image folder path
    video_name = os.path.basename(os.path.normpath(args.image_folder))
    
    # Check if JSON already exists
    bbox_dir = "data/bbox"
    os.makedirs(bbox_dir, exist_ok=True)
    json_path = os.path.join(bbox_dir, f"{video_name}.json")
    
    if os.path.exists(json_path):
        print(f"JSON file already exists for {video_name} at {json_path}. Skipping inference.")
        return
    
    # Load model
    model = load_model(args.config_file, args.checkpoint_path, cpu_only=args.cpu_only)
    
    # Get all image files
    image_files = []
    valid_extensions = ['.jpg', '.jpeg', '.png']
    for ext in valid_extensions:
        image_files.extend(glob.glob(os.path.join(args.image_folder, f"*{ext}")))
    
    # Sort image files to maintain order
    image_files = sorted(image_files)
    
    # Create output directory for inference images
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Dictionary to store all detections
    all_detections = {}
    
    # Process each image
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            # Get output filename
            image_name = os.path.basename(image_path)
            base_name = os.path.splitext(image_name)[0]
            output_name = f"{base_name}_pred.jpg"
            save_path = os.path.join(args.output_dir, output_name)
            
            # Load and process image
            image_pil, image = load_image(image_path)
            
            # Run model
            boxes_filt, pred_phrases, confidence_scores = get_grounding_output(
                model=model,
                image=image,
                caption=args.text_prompt,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                cpu_only=args.cpu_only
            )
            
            # Convert detections to JSON format
            json_boxes = convert_boxes_to_json(boxes_filt, confidence_scores, image_pil.size)
            all_detections[base_name] = json_boxes
            
            # Visualize predictions
            pred_dict = {
                "boxes": boxes_filt,
                "size": [image_pil.size[1], image_pil.size[0]],  # H,W
                "labels": pred_phrases,
            }
            
            # Draw boxes and save
            image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
            image_with_box.save(save_path)
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue
    
    # Save all detections to JSON file
    video_name = os.path.basename(os.path.normpath(args.image_folder))
    json_path = os.path.join(bbox_dir, f"{video_name}_boxes.json")
    with open(json_path, 'w') as f:
        json.dump(all_detections, f, indent=4)
    
    print(f"Saved bbox coordinates to {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounding DINO for folder", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, required=True,
                      help="path to config file")
    parser.add_argument("--checkpoint_path", "-p", type=str, required=True,
                      help="path to checkpoint file")
    parser.add_argument("--image_folder", "-i", type=str, required=True,
                      help="path to image folder")
    parser.add_argument("--text_prompt", "-t", type=str, required=True,
                      help="text prompt")
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                      help="output directory")
    parser.add_argument("--box_threshold", type=float, default=0.3,
                      help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25,
                      help="text threshold")
    parser.add_argument("--cpu-only", action="store_true",
                      help="running on cpu only!")
    
    args = parser.parse_args()
    
    process_images_in_folder(args)