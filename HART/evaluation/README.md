# Usage

> The evaluation toolkit for HART model is basically the same as Infinity

## ImageReward
[ImageReward](https://github.com/THUDM/ImageReward) is a metric for evaluating the human preference score of generated images. It learns human preference through fine-tuning CLIP model with 137K human ranked image pairs.
```shell
out_dir=${out_dir_root}/image_reward_${sub_fix}
infer_eval_image_reward
```

## HPS v2.1
[HPSv2.1](https://github.com/tgxs002/HPSv2) is a metric for evaluating the human preference score of generated images. It learns human preference through fine-tuning CLIP model with 798K human ranked image pairs. The human ranked image pairs are from human experts.
```shell
out_dir=${out_dir_root}/hpsv21_${sub_fix}
infer_eval_hpsv21
```

## GenEval
[GenEval](https://github.com/djghosh13/geneval) is an object-focused framework for evaluating Text-to-Image alignment.
```shell
rewrite_prompt=0
out_dir=${out_dir_root}/gen_eval_${sub_fix}
test_gen_eval
```

## FID
For testing FID, you need provide a jsonl file which contains text prompts and ground truth images. We highly recommand the number of examples in the jsonl file is greater than 20000 since testing FID needs abundant of examples.
```shell
long_caption_fid=1
jsonl_filepath=[jsonl_filepath]
out_dir=${out_dir_root}/val_long_caption_fid_${sub_fix}
rm -rf ${out_dir}
test_fid
```


## MJHQ30K
For testing Validation Loss, you need provide a jsonl folder like the training jsonl folder. Besides, you should specify the noise applying strength for Bitwise Self-Correction to the same strength used in the training phrase.
```shell
python mjhq30k_fid_clip.py
```