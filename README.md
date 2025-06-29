<p align="center">
    <img src="assets/logo.jpg" width="700">
</p>

<div align="center">
**2K resolution image generation with on single 3090 GPU** 🏔️
<img src="assets/teaser.jpg" style="border-radius: 15px">

<h2>
FastVAR: Linear Visual Autoregressive Modeling via Cached Token Pruning (ICCV25)
</h2>

[Hang Guo](https://csguoh.github.io/), [Yawei Li](https://yaweili.bitbucket.io/), [Taolin Zhang](https://github.com/taolinzhang),  [Jiangshan Wang](https://scholar.google.com.hk/citations?user=HoKoCv0AAAAJ&hl=zh-CN&oi=ao), [Tao Dai](https://scholar.google.com.hk/citations?user=MqJNdaAAAAAJ&hl=zh-CN&oi=ao), [Shu-Tao Xia](https://scholar.google.com.hk/citations?hl=zh-CN&user=koAXTXgAAAAJ), [Luca Benini](https://ee.ethz.ch/the-department/people-a-z/person-detail.luca-benini.html)

![visitors](https://visitor-badge.laobi.icu/badge?page_id=cshguo.FastVAR)
[![arXiv](https://img.shields.io/badge/arXiv-2503.23367-b31b1b.svg)](https://arxiv.org/pdf/2503.23367)

</div>

> **Abstract:**  Visual Autoregressive (VAR) modeling has gained popularity for its shift towards next-scale prediction. However, existing VAR paradigms process the entire token map at each scale step, leading to the complexity and runtime scaling dramatically with image resolution. To address this challenge, we propose FastVAR, a post-training acceleration method for efficient resolution scaling with VARs. Our key finding is that the majority of latency arises from the large-scale step where most tokens have already converged. Leveraging this observation, we develop the cached token pruning strategy that only forwards pivotal tokens for scalespecific modeling while using cached tokens from previous scale steps to restore the pruned slots. This significantly reduces the number of forwarded tokens and improves the efficiency at larger resolutions. Experiments show the proposed FastVAR can further speedup FlashAttentionaccelerated VAR by 2.7× with negligible performance drop of <1%. We further extend FastVAR to zero-shot generation of higher resolution images. In particular, FastVAR can generate one 2K image with 15GB memory footprints in 1.5s on a single NVIDIA 3090 GPU. 



⭐If this work is helpful for you, please help star this repo. Thanks!🤗

## ✨ Highlights


1️⃣ **Faster VAR Generation without Perceptual Loss** 

<p align="center">
    <img src="assets/visual.jpg" style="border-radius: 15px">
</p>

2️⃣ **High-resolution Image Generation (even 2K image on single 3090 GPU)**

<p align="center">
    <img src="assets/high_resolution.jpg" style="border-radius: 15px">
</p>


3️⃣ **Promising Resolution Scalibility (almost linear complexity)** 

<p align="center">
    <img src="assets/efficiency.jpg" width="600" style="border-radius: 15px">
</p>


## 📑 Contents

- [News](#news)
- [Pipeline](#pipeline)
- [TODO](#todo)
- [Results](#results)
- [Citation](#cite)

## <a name="news"></a> 🆕 News

- **2025-03-30:** arXiv paper available.
- **2025-04-04:** This repo is released.
- **2025-06-26:** Congrats! Our FastVAR has been accepted by ICCV2025 😊
- **2025-06-29:** We have open sourced all our code.

## <a name="todo"></a> ☑️ TODO

- [x] arXiv version available 
- [x] Release code
- [ ] Further improvements


## <a name="pipeline"></a> 👀 Pipeline

Our FastVAR introduces the **"cached token pruning"** which works on the large-scale steps of the VAR models, which is **training-free** and **generic** for various VAR backbones.

<p align="center">
    <img src="assets/pipeline.jpg" style="border-radius: 15px">
</p>


## <a name="results"></a> 🥇 Results

Our FastVAR can achieve **2.7x** speedup with **<1%** performance drop, even on top of [Flash-attention](https://arxiv.org/abs/2205.14135) accelerated setups. 

Detailed results can be found in the paper.

<details>
<summary>Quantitative Results on the GenEval benchmark(click to expand)</summary>

<p align="center">
  <img width="900" src="assets/results.jpg">
</p>
</details>


<details>
<summary>Quantitative Results on the MJHQ30K benchmark (click to expand)</summary>

<p align="center">
  <img width="500" src="assets/results2.jpg">
</p>
</details>


<details>
<summary>Comparison and combination with FlashAttention (click to expand)</summary>

<p align="center">
  <img width="500" src="assets/flash_attn.jpg">
</p>
</details>



## 🎈Core Algorithm

For learning purpose, we provide the core algorithm of our FastVAR as follows (one may find the complete code in [this line]()). Since our FastVAR is a general technology, other VAR-based models also potentially apply.

```
def masked_previous_scale_cache(cur_x, num_remain, cur_shape):
    B, L, c = cur_x.shape
    mean_x = cur_x.view(B, cur_shape[1], cur_shape[2], -1).permute(0, 3, 1, 2)
    mean_x = torch.nn.functional.adaptive_avg_pool2d(mean_x,(1,1)).permute(0, 2, 3, 1).view(B, 1,c)
    mse_difference = torch.sum((cur_x - mean_x)**2,dim=-1,keepdim=True)
    select_indices = torch.argsort(mse_difference,dim=1,descending=True)
    filted_select_indices=select_indices[:,:num_remain,:]

    def merge(merged_cur_x):
        return torch.gather(merged_cur_x,dim=1,index=filted_select_indices.repeat(1,1,c))

    def unmerge(unmerged_cur_x, unmerged_cache_x, cached_hw=None):
        unmerged_cache_x_ = unmerged_cache_x.view(B, cached_hw[0], cached_hw[1], -1).permute(0, 3, 1, 2)
        unmerged_cache_x_ = torch.nn.functional.interpolate(unmerged_cache_x_, size=(cur_shape[1], cur_shape[2]), mode='area').permute(0, 2, 3, 1).view(B, L, c)
        unmerged_cache_x_.scatter_(dim=1,index=filted_select_indices.repeat(1,1,c),src=unmerged_cur_x)
        return unmerged_cache_x_

    def get_src_tgt_idx():
        return filted_select_indices

    return merge, unmerge, get_src_tgt_idx
```



## 💪Get Started

We apply our FastVAR on two Text-to-Image VAR models, i.e., [Infinity](https://github.com/FoundationVision/Infinity) and [HART](https://github.com/mit-han-lab/hart). The code for the two models can be found in respective folders. For conda environment and related pre-trained LLM/VLM models, we suggest users to refer to the setup in original [Infinity](https://github.com/FoundationVision/Infinity) and [HART](https://github.com/mit-han-lab/hart) repos. In practice, we find both codebase can be compatible to the other. 

### 1. FastVAR for Infinity Acceleration

First cd into the Infinity folder

```
cd ./Infinity
```

Then you can adjust pre-trained Infinity backbone weights and then run text-to-image inference to generate a single image using given user text prompts via

```
python inference.py
```

If you additionally want to reproduce the reported results in our paper, like GenEval, MJHQ30K, HPSv2.1, and image reward, you may refer to the detailed instruction in [this file](https://github.com/csguoh/FastVAR/blob/main/Infinity/evaluation/README.md), which contains all necessary command to run respective experiments.

### 2. FastVAR for HART Acceleration

First cd into the HART folder

```
cd ./HART
```

Then you can run text-to-image generation with the following command.

```
python inference.py --model_path /path/to/model \
   --text_model_path /path/to/Qwen2 \
   --prompt "YOUR_PROMPT" \
   --sample_folder_dir /path/to/save_dir
```

For evaluating HART on common benchmarks, please refer to [this file](https://github.com/csguoh/FastVAR/blob/main/HART/evaluation/README.md), which is basicly similar to Infinity model.



## <a name="cite"></a> 🥰 Citation

Please cite us if our work is useful for your research.

```
@article{guo2025fastvar,
  title={FastVAR: Linear Visual Autoregressive Modeling via Cached Token Pruning},
  author={Guo, Hang and Li, Yawei and Zhang, Taolin and Wang, Jiangshan and Dai, Tao and Xia, Shu-Tao and Benini, Luca},
  journal={arXiv preprint arXiv:2503.23367},
  year={2025}
}
```

## License

Since this work based on the pre-trained VAR models, users should follow the license of the corresponding backbone models like [HART(MIT License)](https://github.com/mit-han-lab/hart) and [Infinity(MIT License)](https://github.com/FoundationVision/Infinity?tab=readme-ov-file). 


## Contact

If you have any questions during your reproduce, feel free to approach me at cshguo@gmail.com
