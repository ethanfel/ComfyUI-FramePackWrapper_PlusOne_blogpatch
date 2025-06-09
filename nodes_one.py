import logging
import math

import torch

logger = logging.getLogger(__name__)

import comfy.latent_formats
import comfy.model_base
import comfy.model_management as mm
from comfy.utils import common_upscale

from .diffusers_helper.memory import move_model_to_device_with_memory_preservation
from .diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from .diffusers_helper.utils import crop_or_pad_yield_mask
from .nodes import HyVideoModel, HyVideoModelConfig

vae_scaling_factor = 0.476986


class FramePackSingleFrameSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("FramePackMODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "start_latent": (
                    "LATENT",
                    {"tooltip": "init Latents to use for image2image"},
                ),
                "steps": ("INT", {"default": 30, "min": 1}),
                "use_teacache": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Use teacache for faster sampling."},
                ),
                "teacache_rel_l1_thresh": (
                    "FLOAT",
                    {
                        "default": 0.15,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "The threshold for the relative L1 loss.",
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.01},
                ),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 10.0, "min": 0.0, "max": 32.0, "step": 0.01},
                ),
                "shift": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.01},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "latent_window_size": (
                    "INT",
                    {
                        "default": 9,
                        "min": 1,
                        "max": 33,
                        "step": 1,
                        "tooltip": "The size of the latent window to use for sampling.",
                    },
                ),
                "gpu_memory_preservation": (
                    "FLOAT",
                    {
                        "default": 6.0,
                        "min": 0.0,
                        "max": 128.0,
                        "step": 0.1,
                        "tooltip": "The amount of GPU memory to preserve.",
                    },
                ),
                "sampler": (["unipc_bh1", "unipc_bh2"], {"default": "unipc_bh1"}),
                "use_kisekaeichi": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable Kisekaeichi mode for style transfer",
                    },
                ),
            },
            "optional": {
                "image_embeds": ("CLIP_VISION_OUTPUT",),
                "initial_samples": (
                    "LATENT",
                    {"tooltip": "init Latents to use for image2image variation"},
                ),
                "denoise_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "reference_latent": (
                    "LATENT",
                    {"tooltip": "Reference image latent for kisekaeichi mode"},
                ),
                "reference_image_embeds": (
                    "CLIP_VISION_OUTPUT",
                    {"tooltip": "Reference image CLIP embeds for kisekaeichi mode"},
                ),
                "target_index": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 8,
                        "step": 1,
                        "tooltip": "Target index for kisekaeichi mode (recommended: 1)",
                    },
                ),
                "history_index": (
                    "INT",
                    {
                        "default": 13,
                        "min": 0,
                        "max": 16,
                        "step": 1,
                        "tooltip": "History index for kisekaeichi mode (recommended: 13)",
                    },
                ),
                "input_mask": (
                    "MASK",
                    {"tooltip": "Input mask for selective application"},
                ),
                "reference_mask": (
                    "MASK",
                    {"tooltip": "Reference mask for selective features"},
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "FramePackWrapper"
    DESCRIPTION = "Single frame sampler with Kisekaeichi (style transfer) support"

    def process(
        self,
        model,
        shift,
        positive,
        negative,
        latent_window_size,
        use_teacache,
        teacache_rel_l1_thresh,
        steps,
        cfg,
        guidance_scale,
        seed,
        sampler,
        gpu_memory_preservation,
        start_latent=None,
        image_embeds=None,
        initial_samples=None,
        denoise_strength=1.0,
        use_kisekaeichi=False,
        reference_latent=None,
        reference_image_embeds=None,
        target_index=1,
        history_index=13,
        input_mask=None,
        reference_mask=None,
    ):
        print("=== 1フレーム推論モード ===")
        if use_kisekaeichi:
            print("Kisekaeichi（着せ替え）モード有効")
            print(f"target_index: {target_index}, history_index: {history_index}")

        transformer = model["transformer"]
        base_dtype = model["dtype"]
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()

        # Latent処理
        if start_latent is not None:
            start_latent = start_latent["samples"] * vae_scaling_factor
        if initial_samples is not None:
            initial_samples = initial_samples["samples"] * vae_scaling_factor
        if use_kisekaeichi and reference_latent is not None:
            reference_latent = reference_latent["samples"] * vae_scaling_factor
            print(f"参照画像latent: {reference_latent.shape}")

        print("start_latent", start_latent.shape)
        B, C, T, H, W = start_latent.shape

        # 画像エンベッディング処理
        if image_embeds is not None:
            start_image_encoder_last_hidden_state = image_embeds[
                "last_hidden_state"
            ].to(device, base_dtype)
        else:
            start_image_encoder_last_hidden_state = None

        if use_kisekaeichi and reference_image_embeds is not None:
            reference_image_encoder_last_hidden_state = reference_image_embeds[
                "last_hidden_state"
            ].to(device, base_dtype)
            print("参照画像のCLIP embeddingを設定しました")
        else:
            reference_image_encoder_last_hidden_state = None

        # テキストエンベッディング処理
        llama_vec = positive[0][0].to(device, base_dtype)
        clip_l_pooler = positive[0][1]["pooled_output"].to(device, base_dtype)

        if not math.isclose(cfg, 1.0):
            llama_vec_n = negative[0][0].to(device, base_dtype)
            clip_l_pooler_n = negative[0][1]["pooled_output"].to(device, base_dtype)
        else:
            llama_vec_n = torch.zeros_like(llama_vec, device=device)
            clip_l_pooler_n = torch.zeros_like(clip_l_pooler, device=device)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(
            llama_vec_n, length=512
        )

        # シード設定
        rnd = torch.Generator("cpu").manual_seed(seed)

        # === 一つ目のコードに完全準拠した設定 ===

        # 1フレームモード固定設定
        sample_num_frames = 1
        total_latent_sections = 1
        latent_padding = 0
        latent_padding_size = latent_padding * latent_window_size  # 0

        # 一つ目のコードと同じインデックス構造
        indices = torch.arange(
            0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])
        ).unsqueeze(0)
        split_sizes = [1, latent_padding_size, latent_window_size, 1, 2, 16]

        # latent_padding_sizeが0の場合の分割（一つ目のコードと完全同一）
        if latent_padding_size == 0:
            clean_latent_indices_pre = indices[:, 0:1]
            latent_indices = indices[:, 1 : 1 + latent_window_size]
            clean_latent_indices_post = indices[
                :, 1 + latent_window_size : 2 + latent_window_size
            ]
            clean_latent_2x_indices = indices[
                :, 2 + latent_window_size : 4 + latent_window_size
            ]
            clean_latent_4x_indices = indices[
                :, 4 + latent_window_size : 20 + latent_window_size
            ]
            blank_indices = torch.empty((1, 0), dtype=torch.long)
        else:
            (
                clean_latent_indices_pre,
                blank_indices,
                latent_indices,
                clean_latent_indices_post,
                clean_latent_2x_indices,
                clean_latent_4x_indices,
            ) = indices.split(split_sizes, dim=1)

        # 一つ目のコードの重要な処理：1フレームモード時のone_frame_inference処理
        if sample_num_frames == 1:
            if use_kisekaeichi and reference_latent is not None:
                print("=== Kisekaeichi モード設定（完全版） ===")

                # 一つ目のコードのone_frame_inference処理を完全再現
                one_frame_inference = set()
                one_frame_inference.add(f"target_index={target_index}")
                one_frame_inference.add(f"history_index={history_index}")

                # 公式実装に従った処理
                latent_indices = indices[:, -1:]  # デフォルトは最後のフレーム

                # パラメータ解析と処理（一つ目のコードと同じ）
                for one_frame_param in one_frame_inference:
                    if one_frame_param.startswith("target_index="):
                        target_idx = int(one_frame_param.split("=")[1])
                        latent_indices[:, 0] = target_idx
                        print(f"latent_indices設定: target_index={target_idx}")

                    elif one_frame_param.startswith("history_index="):
                        history_idx = int(one_frame_param.split("=")[1])
                        clean_latent_indices_post[:, 0] = history_idx
                        print(
                            f"clean_latent_indices_post設定: history_index={history_idx}"
                        )

                # history_latentsのダミーを作成（一つ目のコードと同じ構造）
                history_latents = torch.zeros(
                    size=(1, 16, 1 + 2 + 16, H, W), dtype=torch.float32, device="cpu"
                )

                # clean_latents_preの設定（入力画像）
                clean_latents_pre = start_latent.to(history_latents.dtype).to(
                    history_latents.device
                )
                if len(clean_latents_pre.shape) < 5:
                    clean_latents_pre = clean_latents_pre.unsqueeze(2)

                # マスクの適用（入力画像）
                if input_mask is not None:
                    print("入力画像マスクを適用中...")
                    try:
                        height_latent, width_latent = clean_latents_pre.shape[-2:]

                        if isinstance(input_mask, torch.Tensor):
                            input_mask_tensor = input_mask
                        else:
                            input_mask_tensor = torch.from_numpy(input_mask)

                        input_mask_resized = (
                            common_upscale(
                                input_mask_tensor.unsqueeze(0).unsqueeze(0),
                                width_latent,
                                height_latent,
                                "bilinear",
                                "center",
                            )
                            .squeeze(0)
                            .squeeze(0)
                        )
                        input_mask_resized = input_mask_resized.to(
                            clean_latents_pre.device
                        )[None, None, :, :]
                        clean_latents_pre = clean_latents_pre * input_mask_resized
                        print("入力画像マスクを適用しました")
                    except Exception as e:
                        print(f"入力マスク適用エラー: {e}")

                # clean_latents_postの設定（参照画像）
                clean_latents_post = (
                    reference_latent[:, :, 0:1, :, :]
                    .to(history_latents.dtype)
                    .to(history_latents.device)
                )

                # マスクの適用（参照画像）
                if reference_mask is not None:
                    print("参照画像マスクを適用中...")
                    try:
                        height_latent, width_latent = clean_latents_post.shape[-2:]

                        if isinstance(reference_mask, torch.Tensor):
                            reference_mask_tensor = reference_mask
                        else:
                            reference_mask_tensor = torch.from_numpy(reference_mask)

                        reference_mask_resized = (
                            common_upscale(
                                reference_mask_tensor.unsqueeze(0).unsqueeze(0),
                                width_latent,
                                height_latent,
                                "bilinear",
                                "center",
                            )
                            .squeeze(0)
                            .squeeze(0)
                        )
                        reference_mask_resized = reference_mask_resized.to(
                            clean_latents_post.device
                        )[None, None, :, :]
                        clean_latents_post = clean_latents_post * reference_mask_resized
                        print("参照画像マスクを適用しました")
                    except Exception as e:
                        print(f"参照マスク適用エラー: {e}")

                # clean_latentsを結合
                clean_latents = torch.cat(
                    [clean_latents_pre, clean_latents_post], dim=2
                )

                # clean_latent_indicesを結合
                clean_latent_indices = torch.cat(
                    [clean_latent_indices_pre, clean_latent_indices_post], dim=1
                )

                clean_latents_2x_param = None
                clean_latents_4x_param = None
                clean_latent_2x_indices = None
                clean_latent_4x_indices = None

                # 2x, 4x latentsの設定も無効化
                clean_latents_2x = None
                clean_latents_4x = None

                print("Kisekaeichi: 2x/4xインデックスを無効化しました")

                # 画像エンベッディングの処理（両方を活用）
                if (
                    reference_image_encoder_last_hidden_state is not None
                    and start_image_encoder_last_hidden_state is not None
                ):
                    # 重み付き平均で統合
                    ref_weight = 0.3  # 参照画像の重み
                    input_weight = 1.0 - ref_weight
                    image_encoder_last_hidden_state = (
                        start_image_encoder_last_hidden_state * input_weight
                        + reference_image_encoder_last_hidden_state * ref_weight
                    )
                    print(
                        f"画像エンベッディングを統合 (入力:{input_weight:.2f}, 参照:{ref_weight:.2f})"
                    )
                elif reference_image_encoder_last_hidden_state is not None:
                    image_encoder_last_hidden_state = (
                        reference_image_encoder_last_hidden_state
                    )
                else:
                    image_encoder_last_hidden_state = (
                        start_image_encoder_last_hidden_state
                    )

                print(f"Kisekaeichi設定完了:")
                print(f"  - clean_latents.shape: {clean_latents.shape} (入力+参照)")
                print(f"  - latent_indices: {latent_indices}")
                print(f"  - clean_latent_indices: {clean_latent_indices}")
                print(f"  - sample_num_frames: {sample_num_frames}")
                print(f"  - 2x/4x無効化: True")

            else:
                # 通常モード（参照画像なし）
                all_indices = torch.arange(0, latent_window_size).unsqueeze(0)
                latent_indices = all_indices[:, -1:]

                clean_latents_pre = start_latent.to(torch.float32).cpu()
                if len(clean_latents_pre.shape) < 5:
                    clean_latents_pre = clean_latents_pre.unsqueeze(2)

                clean_latents_post = torch.zeros_like(clean_latents_pre)
                clean_latents = torch.cat(
                    [clean_latents_pre, clean_latents_post], dim=2
                )
                clean_latent_indices = torch.cat(
                    [clean_latent_indices_pre, clean_latent_indices_post], dim=1
                )

                # 通常モードでのインデックス調整
                clean_latent_indices = torch.tensor(
                    [[0]],
                    dtype=clean_latent_indices.dtype,
                    device=clean_latent_indices.device,
                )
                clean_latents = clean_latents[:, :, :1, :, :]

                clean_latents_2x_param = None
                clean_latents_4x_param = None
                clean_latent_2x_indices = None
                clean_latent_4x_indices = None

                # 2x, 4x latentsの設定も無効化
                clean_latents_2x = None
                clean_latents_4x = None

                print("Kisekaeichi: 2x/4xインデックスを無効化しました")

                image_encoder_last_hidden_state = start_image_encoder_last_hidden_state

                print("通常モード設定:")
                print(f"  - clean_latents.shape: {clean_latents.shape}")
                print(f"  - latent_indices: {latent_indices}")
                print(f"  - clean_latent_indices: {clean_latent_indices}")

        # 初期サンプルの処理
        input_init_latents = None
        if initial_samples is not None:
            input_init_latents = initial_samples[:, :, 0:1, :, :].to(device)
            print("初期サンプルを設定しました")

        # ComfyUI用のセットアップ
        comfy_model = HyVideoModel(
            HyVideoModelConfig(base_dtype),
            model_type=comfy.model_base.ModelType.FLOW,
            device=device,
        )
        patcher = comfy.model_patcher.ModelPatcher(
            comfy_model, device, torch.device("cpu")
        )
        from latent_preview import prepare_callback

        callback = prepare_callback(patcher, steps)

        # モデルをGPUに移動
        move_model_to_device_with_memory_preservation(
            transformer,
            target_device=device,
            preserved_memory_gb=gpu_memory_preservation,
        )

        # TeaCacheの設定
        if use_teacache:
            transformer.initialize_teacache(
                enable_teacache=True,
                num_steps=steps,
                rel_l1_thresh=teacache_rel_l1_thresh,
            )
        else:
            transformer.initialize_teacache(enable_teacache=False)

        print("=== サンプリング開始 ===")
        print(f"sample_num_frames: {sample_num_frames}")
        print(f"clean_latents使用フレーム数: {clean_latents.shape[2]}")
        print(f"clean_latent_2x_indices: {clean_latent_2x_indices}")
        print(f"clean_latent_4x_indices: {clean_latent_4x_indices}")

        with torch.autocast(
            device_type=mm.get_autocast_device(device), dtype=base_dtype, enabled=True
        ):
            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler=sampler,
                initial_latent=input_init_latents,
                strength=denoise_strength,
                width=W * 8,
                height=H * 8,
                frames=sample_num_frames,  # 1フレーム固定
                real_guidance_scale=cfg,
                distilled_guidance_scale=guidance_scale,
                guidance_rescale=0,
                shift=shift if shift != 0 else None,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=device,
                dtype=base_dtype,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )

        # クリーンアップ
        transformer.to(offload_device)
        mm.soft_empty_cache()

        # 出力処理
        mode_info = (
            "Kisekaeichi（latent+エンベッディング）" if use_kisekaeichi else "通常"
        )
        print(
            f"=== 1フレーム生成完了 ({mode_info}モード): {generated_latents.shape} ==="
        )

        return ({"samples": generated_latents / vae_scaling_factor},)


NODE_CLASS_MAPPINGS = {
    "FramePackSingleFrameSampler": FramePackSingleFrameSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FramePackSingleFrameSampler": "FramePack Single Frame Sampler",
}
