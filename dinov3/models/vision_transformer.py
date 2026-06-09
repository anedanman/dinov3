# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn.init
from torch import Tensor, nn

from dinov3.layers import (
    LayerScale,
    Mlp,
    PatchClsSeparateRegisterBudgetAttention,
    PatchEmbed,
    RegisterSlotAttention,
    RMSNorm,
    RopePositionEmbedding,
    SelfAttention,
    SelfAttentionBlock,
    SwiGLUFFN,
)
from dinov3.layers.attention import extract_register_attention_maps
from dinov3.utils import named_apply

logger = logging.getLogger("dinov3")

ffn_layer_dict = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}

norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": RMSNorm,
}

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def init_weights_vit(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if hasattr(module, "bias_mask") and module.bias_mask is not None:
            o = module.out_features
            module.bias_mask.fill_(1)
            module.bias_mask[o // 3 : 2 * o // 3].fill_(0)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        module.reset_parameters()
    if isinstance(module, RMSNorm):
        module.reset_parameters()


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = None,
        pos_embed_rope_dtype: str = "bf16",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float | None = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = False,
        register_attn_type: str = "standard",
        slot_mode: str = "slot",
        register_attn_exclude_cls: bool = True,
        patch_cls_attn_type: str = "standard",
        register_init: str = "learned",
        register_gaussian_std_init: float = 0.02,
        device: Any | None = None,
        **ignored_kwargs,
    ):
        super().__init__()
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        norm_layer_cls = norm_layer_dict[norm_layer]

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )

        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.n_storage_tokens = n_storage_tokens
        assert register_init in ("learned", "gaussian"), f"unknown register_init={register_init}"
        assert register_gaussian_std_init > 0, "register_gaussian_std_init must be positive"
        self.register_init = register_init
        self.register_gaussian_std_init = register_gaussian_std_init
        if self.n_storage_tokens > 0:
            n_storage_token_params = 1 if self.register_init == "gaussian" else n_storage_tokens
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_token_params, embed_dim, device=device))
            if self.register_init == "gaussian":
                self.storage_tokens_log_sigma = nn.Parameter(
                    torch.empty(1, n_storage_token_params, embed_dim, device=device)
                )
        logger.info(f"using base={pos_embed_rope_base} for rope new")
        logger.info(f"using min_period={pos_embed_rope_min_period} for rope new")
        logger.info(f"using max_period={pos_embed_rope_max_period} for rope new")
        logger.info(f"using normalize_coords={pos_embed_rope_normalize_coords} for rope new")
        logger.info(f"using shift_coords={pos_embed_rope_shift_coords} for rope new")
        logger.info(f"using rescale_coords={pos_embed_rope_rescale_coords} for rope new")
        logger.info(f"using jitter_coords={pos_embed_rope_jitter_coords} for rope new")
        logger.info(f"using dtype={pos_embed_rope_dtype} for rope new")
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
            device=device,
        )
        logger.info(f"using {ffn_layer} layer as FFN")
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth

        # Register-token attention behaviour.
        assert register_attn_type in ("standard", "slot"), f"unknown register_attn_type={register_attn_type}"
        assert patch_cls_attn_type in ("standard", "separate_register_budget"), (
            f"unknown patch_cls_attn_type={patch_cls_attn_type}"
        )
        self.register_attn_type = register_attn_type
        self.slot_mode = slot_mode
        self.register_attn_exclude_cls = register_attn_exclude_cls
        self.patch_cls_attn_type = patch_cls_attn_type
        if register_attn_type == "slot":
            assert n_storage_tokens > 0, "register_attn_type='slot' requires n_storage_tokens > 0"
            logger.info(
                "using SLOT register attention "
                f"(mode={slot_mode}, exclude_cls={register_attn_exclude_cls}, "
                f"patch_cls_attn_type={patch_cls_attn_type}) with {n_storage_tokens} registers"
            )
            attn_class = partial(
                RegisterSlotAttention,
                n_storage_tokens=n_storage_tokens,
                slot_mode=slot_mode,
                exclude_cls=register_attn_exclude_cls,
                patch_cls_attn_type=patch_cls_attn_type,
            )
        elif patch_cls_attn_type == "separate_register_budget":
            assert n_storage_tokens > 0, "patch_cls_attn_type='separate_register_budget' requires n_storage_tokens > 0"
            logger.info(
                "using separate register attention budget for CLS/patch rows "
                f"with {n_storage_tokens} registers"
            )
            attn_class = partial(
                PatchClsSeparateRegisterBudgetAttention,
                n_storage_tokens=n_storage_tokens,
            )
        else:
            attn_class = SelfAttention

        blocks_list = [
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio_sequence[i],
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=drop_path_rate,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init,
                attn_class=attn_class,
                mask_k_bias=mask_k_bias,
                device=device,
            )
            for i in range(depth)
        ]

        self.chunked_blocks = False
        self.blocks = nn.ModuleList(blocks_list)

        # This norm is applied to everything, or when untying, to patch and mask tokens.
        self.norm = norm_layer_cls(embed_dim)

        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        if untie_cls_and_patch_norms:
            # When untying, this norm is applied to CLS tokens and registers.
            self.cls_norm = norm_layer_cls(embed_dim)
        else:
            self.cls_norm = None

        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        if untie_global_and_local_cls_norm:
            # When untying, this norm is applied to local CLS tokens and registers.
            # This norm is never used during eval.
            self.local_cls_norm = norm_layer_cls(embed_dim)
        else:
            self.local_cls_norm = None
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))

    def init_weights(self):
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            if self.register_init == "gaussian":
                nn.init.xavier_uniform_(self.storage_tokens)
                nn.init.xavier_uniform_(self.storage_tokens_log_sigma)
            else:
                nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(init_weights_vit, self)

    def prepare_tokens_with_masks(self, x: Tensor, masks=None) -> Tuple[Tensor, Tuple[int]]:
        x = self.patch_embed(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token
        if self.n_storage_tokens > 0:
            if self.register_init == "gaussian":
                mu = self.storage_tokens.expand(B, self.n_storage_tokens, -1)
                sigma = self.storage_tokens_log_sigma.exp().expand(B, self.n_storage_tokens, -1)
                eps = torch.randn(mu.shape, dtype=mu.dtype, device=mu.device)
                storage_tokens = mu + eps * sigma
            else:
                storage_tokens = self.storage_tokens.expand(B, -1, -1)
        else:
            storage_tokens = torch.empty(
                B,
                0,
                cls_token.shape[-1],
                dtype=cls_token.dtype,
                device=cls_token.device,
            )

        x = torch.cat(
            [
                cls_token.expand(B, -1, -1),
                storage_tokens,
                x,
            ],
            dim=1,
        )

        return x, (H, W)

    def forward_features_list(self, x_list: List[Tensor], masks_list: List[Tensor]) -> List[Dict[str, Tensor]]:
        x = []
        rope = []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(hw_tuple)
        for _, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = [self.rope_embed(H=H, W=W) for H, W in rope]
            else:
                rope_sincos = [None for r in rope]
            x = blk(x, rope_sincos)
        all_x = x
        output = []
        for idx, (x, masks) in enumerate(zip(all_x, masks_list)):
            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                if self.untie_global_and_local_cls_norm and self.training and idx == 1:
                    # Assume second entry of list corresponds to local crops.
                    # We only ever apply this during training.
                    x_norm_cls_reg = self.local_cls_norm(x[:, : self.n_storage_tokens + 1])
                elif self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(x[:, : self.n_storage_tokens + 1])
                else:
                    x_norm_cls_reg = self.norm(x[:, : self.n_storage_tokens + 1])
                x_norm_patch = self.norm(x[:, self.n_storage_tokens + 1 :])
            else:
                x_norm = self.norm(x)
                x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + 1]
                x_norm_patch = x_norm[:, self.n_storage_tokens + 1 :]
            output.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[:, 0],
                    "x_storage_tokens": x_norm_cls_reg[:, 1:],
                    "x_norm_patchtokens": x_norm_patch,
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x: Tensor | List[Tensor], masks: Optional[Tensor] = None) -> List[Dict[str, Tensor]]:
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks])[0]
        else:
            return self.forward_features_list(x, masks)

    def _get_intermediate_layers_not_chunked(self, x: Tensor, n: int = 1) -> List[Tensor]:
        x, (H, W) = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = self.rope_embed(H=H, W=W)
            else:
                rope_sincos = None
            x = blk(x, rope_sincos)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        *,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs_normed = []
            for out in outputs:
                if self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(out[:, : self.n_storage_tokens + 1])
                    x_norm_patch = self.norm(out[:, self.n_storage_tokens + 1 :])
                    outputs_normed.append(torch.cat((x_norm_cls_reg, x_norm_patch), dim=1))
                else:
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed
        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1 : self.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1 :] for out in outputs]
        if reshape:
            B, _, h, w = x.shape
            outputs = [
                out.reshape(B, h // self.patch_size, w // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        elif return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        elif not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        elif return_class_token and return_extra_tokens:
            return tuple(zip(outputs, class_tokens, extra_tokens))

    def _normalize_attention_layers(self, layers: Optional[Sequence[int]] = None) -> List[int]:
        if layers is None:
            return list(range(self.n_blocks))
        out = sorted({int(layer) % self.n_blocks for layer in layers})
        assert len(out) > 0, "at least one attention layer must be requested"
        return out

    @torch.no_grad()
    def get_register_attention_layers(
        self,
        x: Tensor,
        layers: Optional[Sequence[int]] = None,
        directions: Sequence[str] = ("register_to_patch", "patch_to_register"),
    ) -> Dict[str, Any]:
        """Collect raw per-head register attention maps at selected blocks.

        Returns a dict with ``spatial_size``, ``layers`` and one entry per
        direction. Direction entries contain:
          * masks: [L, B, heads, R, H*W]
          * cls:   [L, B, heads, H*W]
        """
        assert self.n_storage_tokens > 0, "no register tokens to visualize"
        directions = tuple(directions)
        for direction in directions:
            assert direction in ("register_to_patch", "patch_to_register"), f"unknown direction={direction}"
        target_layers = self._normalize_attention_layers(layers)
        target_set = set(target_layers)
        last_needed = max(target_set)

        x, (H, W) = self.prepare_tokens_with_masks(x)
        masks_by_direction = {direction: [] for direction in directions}
        cls_by_direction = {direction: [] for direction in directions}
        seen_layers = []
        for i, blk in enumerate(self.blocks):
            rope = self.rope_embed(H=H, W=W) if self.rope_embed is not None else None
            if i in target_set:
                normed = blk.norm1(x)
                qkv = blk.attn.qkv(normed)
                for direction in directions:
                    masks, cls_map = extract_register_attention_maps(
                        qkv,
                        num_heads=self.num_heads,
                        n_storage_tokens=self.n_storage_tokens,
                        scale=blk.attn.scale,
                        rope=rope,
                        apply_rope_fn=blk.attn.apply_rope,
                        attn_type=self.register_attn_type,
                        slot_renorm=(self.slot_mode == "slot"),
                        slot_exclude_cls=self.register_attn_exclude_cls,
                        patch_cls_attn_type=self.patch_cls_attn_type,
                        direction=direction,
                    )
                    masks_by_direction[direction].append(masks)
                    cls_by_direction[direction].append(cls_map)
                seen_layers.append(i)
                if i == last_needed:
                    break
            x = blk(x, rope)
        assert seen_layers == target_layers, f"only collected layers {seen_layers}, expected {target_layers}"
        out: Dict[str, Any] = {"spatial_size": (H, W), "layers": seen_layers}
        for direction in directions:
            out[direction] = {
                "masks": torch.stack(masks_by_direction[direction], dim=0),
                "cls": torch.stack(cls_by_direction[direction], dim=0),
            }
        return out

    @torch.no_grad()
    def get_register_attention_maps(
        self,
        x: Tensor,
        layer: int = -1,
        direction: str = "register_to_patch",
        layer_reduce: str = "select",
        head_reduce: str = "mean",
    ) -> Tuple[Tensor, Tensor]:
        """Register attention masks and CLS map for one direction.

        ``layer_reduce`` is ``select``, ``mean`` or ``second_half``.
        ``head_reduce`` is ``mean`` or ``none``. Mean-head output shapes are
        ``[B, R, H, W]`` and ``[B, H, W]``. Per-head output shapes are
        ``[B, heads, R, H, W]`` and ``[B, heads, H, W]``.
        """
        assert layer_reduce in ("select", "mean", "second_half"), f"unknown layer_reduce={layer_reduce}"
        assert head_reduce in ("mean", "none"), f"unknown head_reduce={head_reduce}"
        layers = None if layer_reduce in ("mean", "second_half") else [layer]
        collection = self.get_register_attention_layers(x, layers=layers, directions=(direction,))
        H, W = collection["spatial_size"]
        masks = collection[direction]["masks"]  # [L,B,h,R,P]
        cls_map = collection[direction]["cls"]  # [L,B,h,P]
        if layer_reduce == "mean":
            masks = masks.mean(dim=0)
            cls_map = cls_map.mean(dim=0)
        elif layer_reduce == "second_half":
            start = masks.shape[0] // 2
            masks = masks[start:].mean(dim=0)
            cls_map = cls_map[start:].mean(dim=0)
        else:
            masks = masks[0]
            cls_map = cls_map[0]
        if head_reduce == "mean":
            masks = masks.mean(dim=1)  # [B,R,P]
            cls_map = cls_map.mean(dim=1)  # [B,P]
            return (
                masks.reshape(masks.shape[0], self.n_storage_tokens, H, W),
                cls_map.reshape(cls_map.shape[0], H, W),
            )
        return (
            masks.reshape(masks.shape[0], self.num_heads, self.n_storage_tokens, H, W),
            cls_map.reshape(cls_map.shape[0], self.num_heads, H, W),
        )

    @torch.no_grad()
    def get_register_patch_attention(self, x: Tensor, layer: int = -1) -> Tensor:
        """Register-token -> patch attention maps for visualization / MBO masks.

        Runs the backbone and, at the requested block (``layer``, default last),
        returns the attention each register token places on the patch grid,
        averaged over heads, reshaped to the spatial grid.

        Returns: [B, R, H, W] attention maps (R = n_storage_tokens).
        """
        attn, _ = self.get_register_attention_maps(
            x,
            layer=layer,
            direction="register_to_patch",
            layer_reduce="select",
            head_reduce="mean",
        )
        return attn

    def forward(self, *args, is_training: bool = False, **kwargs) -> List[Dict[str, Tensor]] | Tensor:
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def vit_small(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_so400m(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1152,
        depth=27,
        num_heads=18,
        ffn_ratio=3.777777778,
        **kwargs,
    )
    return model


def vit_huge2(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1280,
        depth=32,
        num_heads=20,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_7b(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=4096,
        depth=40,
        num_heads=32,
        ffn_ratio=3,
        **kwargs,
    )
    return model
