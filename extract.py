import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForImageTextRetrieval, AutoProcessor
import json
from decord import VideoReader
from decord import cpu
import numpy as np
import os
import argparse


# extract frame per second and give a blip image-text matching score of every sample frame

def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract Video Feature')
    parser.add_argument('--video_path', type=str, default="messi.mp4", required=False, help='Path to the input video file')
    parser.add_argument('--extract_feature_model', type=str, default='blip', help='blip/clip')
    parser.add_argument('--output_file', type=str, default='output', help='Path to output scores and frames')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (cuda or cpu)')
    parser.add_argument('--task', type=str, default='itm', help='blip task type: captioning/itm')  # 新增参数
    parser.add_argument('--blip_weight', type=str, default='blip-itm-large-coco')
    return parser.parse_args()

def main(args):
    # load video and device
    video_path = args.video_path
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    if not os.path.exists(video_path):
        raise OSError(f"Video file {video_path} does not exist")

    # load the model
    if args.extract_feature_model == 'blip':
        if args.task == 'captioning':
            # for caption task
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        elif args.task == 'itm':
            # for image-text matching task
            model = BlipForImageTextRetrieval.from_pretrained(args.blip_weight)
            processor = AutoProcessor.from_pretrained(args.blip_weight)
        else:
            raise ValueError("Unsupported BLIP task. Choose 'captioning' or 'itm'.")
        model.to(device)
    else:
        raise ValueError("Only 'blip' model is supported.")


    # output dir
    out_score_path = os.path.join(args.output_file, 'blip')
    os.makedirs(out_score_path, exist_ok=True)

    scores = []
    fn = []
    score_path = os.path.join(out_score_path, 'scores.json')
    frame_path = os.path.join(out_score_path, 'frames.json')

    # sample the video
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    fps = vr.get_avg_fps()
    frame_indices = range(0, len(vr), int(fps)) 
    
    # question(text)
    text = "Describe the man in the Barcelona football team"

    score = []
    frame_num = []
    for j, frame_idx in enumerate(frame_indices):
        raw_image = np.array(vr[frame_idx].asnumpy(), dtype=np.uint8)
        raw_image = Image.fromarray(raw_image).convert("RGB")

        # caption or itm
        if args.task == 'captioning':
            # caption
            inputs = processor(raw_image, text, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(**inputs)
            scores.append(outputs)

        elif args.task == 'itm':
            # image-text matching
            inputs = processor(raw_image, text, return_tensors="pt").to(device)
            with torch.no_grad():
                itm_output = model(**inputs, use_itm_head=True)
            itm_scores = torch.nn.functional.softmax(itm_output.itm_score, dim=1)
            print(itm_scores[:, 1].item())

        score.append(itm_scores[:, 1].item())
        frame_num.append(frame_idx)
        
    fn.append(frame_num)
    scores.append(score)
    # save
    with open(frame_path, 'w') as f:
        json.dump(fn, f)
    with open(score_path, 'w') as f:
        json.dump(scores, f)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
