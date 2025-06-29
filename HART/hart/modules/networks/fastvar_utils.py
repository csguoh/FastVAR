import torch
from typing import Tuple, Callable
import torch
import math
from typing import Type, Dict, Any, Tuple, Callable


def do_nothing(x: torch.Tensor, *args, **kwargs):
    return x



def masked_previous_scale_cache(cur_x, cur_scale, num_remain):
    B, L, c = cur_x.shape
    cur_x_hw = int(math.sqrt(cur_x.shape[1]))
    mean_x = cur_x.view(B, cur_x_hw, cur_x_hw, -1).permute(0, 3, 1, 2)
    mean_x = torch.nn.functional.adaptive_avg_pool2d(mean_x,(1,1)).permute(0, 2, 3, 1).view(B, 1,c)
    mse_difference = torch.sum((cur_x - mean_x)**2,dim=-1,keepdim=True)
    select_indices = torch.argsort(mse_difference,dim=1,descending=True)
    filted_select_indices=select_indices[:,:num_remain,:]

    def merge(merged_cur_x):
        return torch.gather(merged_cur_x,dim=1,index=filted_select_indices.repeat(1,1,c))

    def unmerge(unmerged_cur_x, unmerged_cache_x):
        tmp_hw = int(math.sqrt(unmerged_cache_x.shape[1]))
        unmerged_cache_x_ = unmerged_cache_x.view(B, tmp_hw, tmp_hw, -1).permute(0, 3, 1, 2)
        unmerged_cache_x_ = torch.nn.functional.interpolate(unmerged_cache_x_, size=(cur_scale, cur_scale), mode='area').permute(0, 2, 3, 1).view(B, L, c)
        unmerged_cache_x_.scatter_(dim=1,index=filted_select_indices.repeat(1,1,c),src=unmerged_cur_x)
        return unmerged_cache_x_

    def get_src_tgt_idx():
        return filted_select_indices

    return merge, unmerge, get_src_tgt_idx




# [1, 2, 3, 4, 5, 7, 9, 12, 16, 21, 27, 36, 48, 64]
def compute_merge(x: torch.Tensor, ratio=0.5, prune_scale_list=[48,64], is_later_layer=False) -> Tuple[Callable, ...]:
    original_h = original_w = int(math.sqrt(x.shape[1]))
    original_tokens = original_h * original_w
    if original_w in prune_scale_list and is_later_layer:
        ratio_hard_code = {48:0.5, 64:0.75}
        ratio = ratio_hard_code[original_w]
        r = int(x.shape[1] * ratio)
        m, u, id_fn = masked_previous_scale_cache(x,original_h,x.shape[1]-r)
    else:
        m, u, id_fn = (do_nothing, do_nothing, do_nothing)

    m_a, u_a = (m, u)

    return m_a, u_a, id_fn  # Okay this is probably not very good
