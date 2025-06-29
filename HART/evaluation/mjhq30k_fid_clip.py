import argparse
import copy
import sys
sys.path.insert(0,'/data2/guohang/HART')
import datetime
import os
import random
import time
import json
import numpy as np
import torch
import torchvision
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
import clip
from tqdm import tqdm
from hart.modules.models.transformer import HARTForT2I
from hart.utils import default_prompts, encode_prompts, llm_system_prompt, safety_check
from cleanfid import fid
from clip_score import clip_score


def compute_clip_score(model, preprocess,image_path, text,device):
    # Load and preprocess the image
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # Tokenize the text
    text = clip.tokenize([text],truncate=True).to(device)

    # Compute the feature vectors
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

    # Normalize the feature vectors
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute the cosine similarity
    similarity = (image_features @ text_features.T).item()
    return similarity


def save_images(sample_imgs, sample_folder_dir, ):
    sample_imgs_np = sample_imgs.mul_(255).cpu().numpy()
    num_imgs = sample_imgs_np.shape[0]
    os.makedirs(os.path.dirname(sample_folder_dir), exist_ok=True)
    for img_idx in range(num_imgs):
        cur_img = sample_imgs_np[img_idx]
        cur_img = cur_img.transpose(1, 2, 0).astype(np.uint8)
        cur_img_store = Image.fromarray(cur_img)
        cur_img_store.save(sample_folder_dir)



def main(args):
    device = torch.device("cuda")

    model = AutoModel.from_pretrained(args.model_path)
    model = model.to(device)
    model.eval()

    if args.use_ema:
        ema_model = copy.deepcopy(model)
        ema_model.load_state_dict(
            torch.load(os.path.join(args.model_path, "ema_model.bin"))
        )

    text_tokenizer = AutoTokenizer.from_pretrained(args.text_model_path)
    text_model = AutoModel.from_pretrained(args.text_model_path).to(device)
    text_model.eval()

    with open("/data2/guohang/dataset/MJHQ30K/meta_data.json") as f:
        meta_data = json.load(f)

    for img_id, data in tqdm(meta_data.items()):
        prompt = data['prompt']
        category = data['category']
        if 'people' not in category:
            continue
        os.makedirs(os.path.join(args.sample_folder_dir, category),exist_ok=True)
        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16, cache_enabled=True):
                (
                    context_tokens,
                    context_mask,
                    context_position_ids,
                    context_tensor,
                ) = encode_prompts(
                    [prompt],
                    text_model,
                    text_tokenizer,
                    args.max_token_length,
                    llm_system_prompt,
                    args.use_llm_system_prompt,
                )

                infer_func = (
                    ema_model.autoregressive_infer_cfg
                    if args.use_ema
                    else model.autoregressive_infer_cfg
                )
                output_imgs = infer_func(
                    B=context_tensor.size(0),
                    label_B=context_tensor,
                    cfg=args.cfg,
                    g_seed=args.seed,
                    more_smooth=args.more_smooth,
                    context_position_ids=context_position_ids,
                    context_mask=context_mask,
                )
        save_images(output_imgs.clone(), os.path.join(args.sample_folder_dir,category, f"{img_id}.png"))



    # test fid
    ref_dir = "/data2/guohang/dataset/MJHQ30K/mjhq30k_imgs/people"
    gen_dir = args.sample_folder_dir
    fid_score = fid.compute_fid(ref_dir, gen_dir)
    print(f'FID score:{fid_score}')



    # Load the CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)

    generated_dir = args.sample_folder_dir
    total_score = 0
    count = 0

    for root, _, files in os.walk(generated_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Ensure you're processing image files
                image_path = os.path.join(root, file)
                image_id = os.path.splitext(file)[0]
                if image_id in meta_data:
                    prompt = meta_data[image_id]["prompt"]
                    score = compute_clip_score(model, preprocess,image_path, prompt,device)
                    total_score += score
                    count += 1
                else:
                    print(f"No prompt found for image {image_id}")

    if count > 0:
        average_clip_score = total_score / count
        print(f"Average CLIP Score: {average_clip_score}. total images {count}")
    else:
        print("No images were processed.")







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="The path to HART model.",
        default="/data2/guohang/pretrained/hart-0.7b-1024px/llm",
    )
    parser.add_argument(
        "--text_model_path",
        type=str,
        help="The path to text model, we employ Qwen2-VL-1.5B-Instruct by default.",
        default="/data2/guohang/pretrained/Qwen2-VL-1.5B-Instruct",
    )
    parser.add_argument(
        "--shield_model_path",
        type=str,
        help="The path to shield model, we employ ShieldGemma-2B by default.",
        default="pretrained_models/shieldgemma-2b",
    )
    parser.add_argument("--prompt", type=str, help="A single prompt.", default="A close-up photo of a person. The subject is a woman. She wore a blue coat with a gray dress underneath. She has blue eyes and blond hair, and wears a pair of earrings. Behind are blur red city buildings and streets.")
    parser.add_argument("--prompt_list", type=list, default=[])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--use_ema", type=bool, default=True)
    parser.add_argument("--max_token_length", type=int, default=300)
    parser.add_argument("--use_llm_system_prompt", type=bool, default=True)
    parser.add_argument(
        "--cfg", type=float, help="Classifier-free guidance scale.", default=4.5
    )
    parser.add_argument(
        "--more_smooth",
        type=bool,
        help="Turn on for more visually smooth samples.",
        default=True,
    )
    parser.add_argument(
        "--sample_folder_dir",
        type=str,
        help="The folder where the image samples are stored",
        default="/data2/guohang/HART/mjhq_samples/",
    )
    parser.add_argument(
        "--store_seperately",
        help="Store image samples in a grid or separately, set to False by default.",
        action="store_true",
    )
    args = parser.parse_args()

    main(args)
