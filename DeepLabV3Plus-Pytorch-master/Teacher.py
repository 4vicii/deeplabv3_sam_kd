from functools import partial
import torch

from segment_anything.modeling import ImageEncoderViT


def build_teacher_vit_h(checkpoint=None):
    return _build_teacher(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )

build_stu = build_teacher_vit_h

def build_teacher_vit_l(checkpoint=None):
    return _build_teacher(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )

def build_teacher_vit_b(checkpoint=None):
    return _build_teacher(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )

teacher_model_registry = {
    "default": build_teacher_vit_b,
    "vit_h": build_teacher_vit_h,
    "vit_l": build_teacher_vit_l,
    "vit_b": build_teacher_vit_b,
}

def _build_teacher(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    teacher = ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        )
    if checkpoint is not None:
        teacher.load_state_dict(checkpoint)
    teacher.eval()
    return teacher
