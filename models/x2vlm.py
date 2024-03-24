# X^2-VLM: All-In-One Pre-trained Model For Vision-Language Tasks (https://arxiv.org/abs/2211.12402)
# Github: https://github.com/zengyan-97/X2-VLM
# Copyright (c) 2022, ByteDance Inc.
# All rights reserved.

# Multi-Grained Vision Language Pre-Training: Aligning Texts with Visual Concepts (https://arxiv.org/abs/2111.08276)
# Github: https://github.com/zengyan-97/X-VLM
# Copyright (c) 2022, ByteDance Inc.
# All rights reserved.

import os
import torch
import torch.nn as nn
import torch.distributed as dist

import copy
from models.utils import read_json

class VanillaConfig(object):
    def __init__(self):
        pass


def load_params_change_prefix(state_dict: dict, prefix: str, new_prefix: str):
    if prefix == new_prefix:
        return state_dict

    state_dict_new = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            k = k.replace(prefix, new_prefix)

        state_dict_new[k] = v

    return state_dict_new


def load_roberta_lm_head(state_dict):
    def _replace(old_key: str, new_key: str):
        if new_key != old_key:
            state_dict[new_key] = state_dict[old_key]
            del state_dict[old_key]

    _replace('lm_head.bias', 'cls.predictions.bias')
    _replace('lm_head.dense.weight', 'cls.predictions.transform.dense.weight')
    _replace('lm_head.dense.bias', 'cls.predictions.transform.dense.bias')
    _replace('lm_head.layer_norm.weight', 'cls.predictions.transform.LayerNorm.weight')
    _replace('lm_head.layer_norm.bias', 'cls.predictions.transform.LayerNorm.bias')
    _replace('lm_head.decoder.weight', 'cls.predictions.decoder.weight')


def rename_tf_layernorm(state_dict):
    for k in list(state_dict.keys()):
        if 'LayerNorm.' in k:
            new_k = k.strip().replace('LayerNorm.beta', 'LayerNorm.bias')
            new_k = new_k.strip().replace('LayerNorm.gamma', 'LayerNorm.weight')
            state_dict[new_k] = state_dict[k]
            if new_k != k:
                del state_dict[k]


def load_params_choose_layers(prefix: str, state_dict: dict, mapper: dict, do_expand=False):
    """
        mapper: {old_layer: new_layer}
    """
    # fixed a bug
    # when mapper is for example {0: 0, 2: 1, 4: 2, 5: 3}
    # in the case, 4 -> 2 -> 1, causes error

    assert len(set(mapper.values())) == len(mapper), f"{set(mapper.values())} != {len(mapper)}"  # no overlap

    k_list = sorted([int(k) for k in mapper.keys()])
    mapper = {k: mapper[k] for k in k_list}

    if not len(mapper):
        return state_dict

    param_sorted = []

    for k in list(state_dict.keys()):
        if k.startswith(prefix):
            i_layer = k[len(prefix)+1:]
            i_layer = int(i_layer.strip().split('.')[0])
            param_sorted.append((k, i_layer))
        else:
            param_sorted.append((k, -1))  # any is ok

    param_sorted = sorted(param_sorted, key=lambda p: p[1])
    param_sorted = [p[0] for p in param_sorted]

    for k in param_sorted:  # must start from lower layers
        if k.startswith(prefix):
            new_k = None
            for i in mapper.keys():
                if k.startswith(f'{prefix}.{i}.'):
                    new_k = k.replace(f'{prefix}.{i}.', f'{prefix}.{mapper[i]}.')
                    break

            if new_k:
                state_dict[new_k] = state_dict[k]

            if (new_k != k) and (not do_expand):
                del state_dict[k]

    return state_dict



class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, rank, world_size):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
            None
        )


allgather = AllGather.apply


def build_mlp(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, input_dim * 2),
        nn.LayerNorm(input_dim * 2),
        nn.GELU(),
        nn.Linear(input_dim * 2, output_dim)
    )


def build_vision_encoder(config, load_params=False, is_masked=False):
    """
    Args:
        load_params: False when building fine-tuning models
    """
    num_patches = (config['image_res'] // config['patch_size']) ** 2

    if config.get('use_clip_vit', False):  # good performance, but only base model available
        from models.clip_vit import CLIPVisionTransformer, interpolate_pos_embed

        vision_config = read_json(config['vision_config'])
        assert config['patch_size'] == vision_config['patch_size']
        vision_width = vision_config['vision_width']

        vision_encoder = CLIPVisionTransformer(image_size=config['image_res'], patch_size=vision_config['patch_size'],
                                               hidden_size=vision_config['vision_width'],
                                               hidden_act=vision_config['hidden_act'],
                                               num_attention_heads=vision_config['num_attention_heads'],
                                               attention_dropout=vision_config['attention_dropout'],
                                               intermediate_size=vision_config['intermediate_size'],
                                               num_hidden_layers=vision_config['num_hidden_layers'],
                                               local_attn_depth=vision_config['local_attn_depth'])

        if load_params:
            # download from https://huggingface.co/openai/clip-vit-base-patch16/tree/main
            state_dict_orig = torch.load(vision_config['ckpt'], map_location="cpu")
            state_dict = {}
            for k, v in state_dict_orig.items():
                if k.startswith('vision_model.'):
                    k = k[13:]
                    if k.startswith('embeddings.'):
                        k = k[11:]
                        k = k.replace('patch_embedding.weight', 'patch_embed.weight')
                        k = k.replace('position_embedding.weight', 'pos_embed.weight')

                    if k != 'position_ids':
                        state_dict[k] = v

            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed.weight'].unsqueeze(dim=0),
                                                       num_patches=num_patches, num_extra_tokens=1)
            state_dict['pos_embed.weight'] = pos_embed_reshaped.squeeze(dim=0)

            assert vision_config['num_hidden_layers'] in [6, 12], "param initialization not implemented"
            if vision_config['num_hidden_layers'] == 6:
                mapper = {1: 0, 3: 1, 5: 2, 7: 3, 9: 4, 11: 5}
                load_params_choose_layers('encoder.layers', state_dict, mapper)

    elif config.get('use_swin', False):
        from models.swin_transformer import SwinTransformer

        vision_config = read_json(config['vision_config'])
        assert config['image_res'] == vision_config['image_res']
        assert config['patch_size'] == 32
        vision_width = vision_config['vision_width']

        vision_encoder = SwinTransformer(img_size=vision_config['image_res'],
                                         patch_size=4,
                                         in_chans=3,
                                         embed_dim=vision_config['embed_dim'],
                                         depths=vision_config['depths'],
                                         num_heads=vision_config['num_heads'],
                                         window_size=vision_config['window_size'],
                                         mlp_ratio=4.,
                                         qkv_bias=True,
                                         drop_rate=0.0,
                                         drop_path_rate=0.1,
                                         ape=False,
                                         patch_norm=True,
                                         use_checkpoint=False, add_cls=config.get('swin_add_cls', True))

        if load_params:
            from models.swin_transformer import load_pretrained_swin
            state_dict = load_pretrained_swin(vision_encoder, vision_config['ckpt'])

    elif config.get('use_beit_v2', False):

        vision_config = read_json(config['vision_config'])
        assert config['patch_size'] == vision_config['patch_size']
        vision_width = vision_config['vision_width']

        if 'base' in config['vision_config']:
            if config['use_mask'] and is_masked:
                from models.beit2 import beit_base_patch16_mask as beit_model
            else:
                from models.beit2 import beit_base_patch16 as beit_model
        elif 'large' in config['vision_config']:
            if config['use_mask'] and is_masked:
                from models.beit2 import beit_large_patch16_mask as beit_model
            else:
                from models.beit2 import beit_large_patch16 as beit_model
        else:
            raise ValueError

        vision_encoder = beit_model(img_size=config['image_res'],
                                    drop_rate=0.0, drop_path_rate=0.1, attn_drop_rate=0.0,
                                    use_mean_pooling=True,
                                    init_scale=0.001,
                                    use_rel_pos_bias=True, use_abs_pos_emb=False,
                                    init_values=0.1, qkv_bias=True, local_attn_depth=config.get('local_attn_depth', -1),
                                    vision_num_hidden_layers=config.get('vision_num_hidden_layers', -1))

        if load_params:
            from models.beit2 import load_pretrained_beit2
            load_pretrained_beit2(vision_encoder, vision_config['ckpt'])

    else:
        raise ValueError

    if load_params and (not config.get('use_beit_v2', False)):
        print("### Load ViT: ", flush=True)
        msg = vision_encoder.load_state_dict(state_dict, strict=False)
        print("missing_keys: ", msg.missing_keys)
        print("unexpected_keys: ", msg.unexpected_keys)

    # set attrs
    vision_encoder.vision_width = vision_width

    return vision_encoder

def load_pretrained(model, ckpt_rpath, config, is_eval=False, load_text=False, use_mlm_loss=False):
    checkpoint = torch.load(ckpt_rpath, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
    if is_eval:
        return state_dict

    print("### Loading pretrained vision encoder", flush=True)
    if config.get('use_clip_vit', False):
        from models.clip_vit import interpolate_pos_embed
        del state_dict['vision_encoder.position_ids']
        num_patches = (config['image_res'] // config['patch_size']) ** 2
        pos_embed_reshaped = interpolate_pos_embed(state_dict['vision_encoder.pos_embed.weight'].unsqueeze(dim=0),
                                                   num_patches=num_patches, num_extra_tokens=1)
        state_dict['vision_encoder.pos_embed.weight'] = pos_embed_reshaped.squeeze(dim=0)

    elif config.get('use_swin', False) or config.get('use_swin_v2', False):
        from models.swin_transformer import load_pretrained_swin

        vision_state_dict = {}
        for k in list(state_dict.keys()):
            if k.startswith('vision_encoder.'):
                vision_state_dict[k[15:]] = state_dict[k]
                del state_dict[k]

        vision_state_dict = load_pretrained_swin(model.vision_encoder, state_dict=vision_state_dict)

        for k in vision_state_dict.keys():
            state_dict['vision_encoder.' + k] = vision_state_dict[k]

    elif config.get('use_beit_v2', False):
        from models.beit2 import interpolate_pos_embed

        vision_state_dict = {}
        for k in list(state_dict.keys()):
            if k.startswith('vision_encoder.'):
                vision_state_dict[k[15:]] = state_dict[k]
                del state_dict[k]

        vision_state_dict = interpolate_pos_embed(model.vision_encoder, vision_state_dict)
        
        if config['use_momentum']:
            print("Init momentum model")
            for k in vision_state_dict.keys():
                state_dict['vision_encoder_m.' + k] = vision_state_dict[k]
        for k in vision_state_dict.keys():
            state_dict['vision_encoder.' + k] = vision_state_dict[k]

    else:
        raise ValueError

    if load_text and not use_mlm_loss:
        print("### Loading pretrained text encoder", flush=True)
        for key in list(state_dict.keys()):
            if key.startswith('text_encoder.') or key.startswith('cross_encoder.'):
                encoder_key = key.replace('roberta.', '').replace('bert.', '').strip()
                
                state_dict[encoder_key] = state_dict[key]
                if encoder_key != key:
                  del state_dict[key]


    if config.get('init_timesformer', False):
        map_dict = {
            "temporal_norm1": "norm1",
            "time_attn": "attn",
            "temporal_norm2": "norm2",
            "temporal_mlp": "mlp",
            "time_gamma_1": "gamma_1",
            "time_gamma_2": "gamma_2"
        }
        for from_key, to_key in map_dict.items():
            for key in list(state_dict.keys()):
                if to_key in key:
                    state_dict[key.replace(to_key, from_key)] = copy.deepcopy(state_dict[key])

    return state_dict

def load_pretrained_vision_tower(ckpt_path, config, is_eval=False):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
    
    from models.beit2 import interpolate_pos_embed

    vision_state_dict = {}
    for k in list(state_dict.keys()):
        if k.startswith('vision_encoder.'):
            vision_state_dict[k[15:]] = state_dict[k]
            del state_dict[k]

    vision_encoder = build_vision_encoder(config)
    vision_state_dict = interpolate_pos_embed(vision_encoder, vision_state_dict)

    vision_encoder.load_state_dict(vision_state_dict, strict=False)

    return vision_encoder
