import os
import os.path as osp
import hashlib
import time
import argparse
import json
import shutil
import glob
import re
import sys
import copy
from PIL import Image
sys.path.insert(0,'/data2/guohang/HART')
import cv2
from tqdm import tqdm
import torch
import numpy as np
from pytorch_lightning import seed_everything
from transformers import (
    AutoModel,
    AutoTokenizer,
)
from hart.modules.models.transformer import HARTForT2I
from hart.utils import default_prompts, encode_prompts, llm_system_prompt, safety_check

if __name__ == '__main__':
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
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--use_ema", type=bool, default=True)
    parser.add_argument("--max_token_length", type=int, default=300)
    parser.add_argument("--use_llm_system_prompt", type=bool, default=True)
    parser.add_argument("--cfg", type=float, help="Classifier-free guidance scale.", default=4.5)
    parser.add_argument("--more_smooth",type=bool, help="Turn on for more visually smooth samples.",default=True,)

    # ----
    parser.add_argument('--outdir', type=str, default='')
    parser.add_argument('--n_samples', type=int, default=4)
    parser.add_argument('--metadata_file', type=str, default='/data2/guohang/HART/evaluation/gen_eval/prompts/evaluation_metadata.jsonl')
    parser.add_argument('--rewrite_prompt', type=int, default=0, choices=[0,1])
    parser.add_argument('--load_rewrite_prompt_cache', type=int, default=0, choices=[0,1])
    args = parser.parse_args()

    
    with open(args.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]


    prompt_rewrite_cache_file = osp.join('/data2/guohang/Infinite/evaluation/gen_eval', 'prompt_rewrite_cache.json')
    if osp.exists(prompt_rewrite_cache_file):
        with open(prompt_rewrite_cache_file, 'r') as f:
            prompt_rewrite_cache = json.load(f)
    else:
        prompt_rewrite_cache = {}


    device = torch.device("cuda")
    model = AutoModel.from_pretrained(args.model_path)
    model = model.to(device)
    model.eval()

    if args.use_ema:
        ema_model = copy.deepcopy(model)
        ema_model.load_state_dict(torch.load(os.path.join(args.model_path, "ema_model.bin")))

    text_tokenizer = AutoTokenizer.from_pretrained(args.text_model_path)
    text_model = AutoModel.from_pretrained(args.text_model_path).to(device)
    text_model.eval()


    for index, metadata in tqdm(enumerate(metadatas)):
        seed_everything(args.seed)
        outpath = os.path.join(args.outdir, f"{index:0>5}")
        os.makedirs(outpath, exist_ok=True)
        prompt = metadata['prompt']
        print(f"Prompt ({index: >3}/{len(metadatas)}): '{prompt}'")

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

        if args.rewrite_prompt:
            old_prompt = prompt
            if args.load_rewrite_prompt_cache and prompt in prompt_rewrite_cache:
                prompt = prompt_rewrite_cache[prompt]
            print(f'old_prompt: {old_prompt}, refined_prompt: {prompt}')

        images = []
        for sample_j in range(args.n_samples):
            print(f"Generating {sample_j+1} of {args.n_samples}, prompt={prompt}")
            t1 = time.time()
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

            t2 = time.time()
            print(f'infer one image takes {t2-t1:.2f}s')
            torch.cuda.empty_cache()
            images.append(output_imgs.clone().float())

        for i, image in enumerate(images):
            save_file = os.path.join(sample_path, f"{i:05}.jpg")
            sample_imgs_np = image.mul_(255).cpu().numpy()
            num_imgs = sample_imgs_np.shape[0]
            for img_idx in range(num_imgs):
                cur_img = sample_imgs_np[img_idx]
                cur_img = cur_img.transpose(1, 2, 0).astype(np.uint8)
                cur_img_store = Image.fromarray(cur_img)
                cur_img_store.save(os.path.join(save_file))
    

