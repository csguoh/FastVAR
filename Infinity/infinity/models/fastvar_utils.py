import torch
from typing import Tuple, Callable
import torch
import math
from typing import Type, Dict, Any, Tuple, Callable


def do_nothing(x: torch.Tensor, *args, **kwargs):
    return x


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




# 1/2 : [... (1, 23, 46), (1, 30, 60), (1, 37, 74), (1, 45, 90), (1, 60, 120)]
# 1.333/1  (1, 36, 27), (1, 48, 36), (1, 60, 45), (1, 72, 54) (1,84,63)
# 2/1:  (1, 46, 23), (1, 60, 30), (1, 74, 37), (1, 90, 45) (1,120,60)
# 1/1 , (13, 32, 32), (15, 40, 40), (17, 48, 48), (21, 64, 64), (1, 84, 84)]
def compute_merge(x: torch.Tensor, prune_scale_list=[32, 40], is_later_layer=False, x_shape=None) -> Tuple[Callable, ...]:
    _, original_h, original_w = x_shape
    original_tokens = original_h * original_w

    if original_w in prune_scale_list and is_later_layer:
        ratio_hard_code = {32:0.4, 40:0.5}
        ratio =ratio_hard_code[original_w]
        r = int(x.shape[1] * ratio)
        m, u, id_fn = masked_previous_scale_cache(x,x.shape[1]-r,x_shape)
    else:
        m, u, id_fn = (do_nothing, do_nothing, do_nothing)

    m_a, u_a = (m, u)

    return m_a, u_a, id_fn  # Okay this is probably not very good


