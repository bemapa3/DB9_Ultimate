"""VAE, sampler (hỗ trợ Flux2Scheduler), Differential Diffusion, ControlNet adapters."""
import torch

import comfy.model_management as mm
import comfy.sample
import comfy.utils
import latent_preview

from . import core


# ----------------------------------------------------------------------------
# VAE
# ----------------------------------------------------------------------------
def vae_encode(vae, img):
    x = img[..., :3]
    try:
        return vae.encode(x)
    except mm.OOM_EXCEPTION:
        raise
    except RuntimeError as e:
        # VAE 4 kênh (RGBA, vd Qwen-Image 2.1) -> thêm alpha = 1. Lỗi khác: báo nguyên gốc
        if "channel" not in str(e).lower():
            raise
        return vae.encode(torch.cat([x, torch.ones_like(x[..., :1])], -1))


def vae_decode(vae, samples):
    d = vae.decode(samples)
    while d.ndim > 4:
        d = d.reshape(-1, *d.shape[-3:])
    return d[..., :3].float().cpu().clamp(0, 1)


# ----------------------------------------------------------------------------
# Conditioning
# ----------------------------------------------------------------------------
def prep_cond(cond, mode, latent=None):
    """strip: bỏ reference_latents (img2img thuần) · tile: gắn latent của chính ô · keep: giữ nguyên."""
    if mode == "keep":
        return cond
    out = []
    for c in cond:
        d = {k: v for k, v in c[1].items() if k != "reference_latents"}
        if mode == "tile" and latent is not None:
            d["reference_latents"] = [latent]
        out.append([c[0], d])
    return out


# ----------------------------------------------------------------------------
# Sampler
# ----------------------------------------------------------------------------
def flux2_sigmas(steps, width, height, denoise):
    """Sigmas giống node Flux2Scheduler (dịch theo độ phân giải), cắt đuôi theo denoise."""
    from comfy_extras.nodes_flux import get_schedule
    denoise = max(1e-3, min(1.0, denoise))
    total = steps if denoise >= 0.9999 else max(steps, int(round(steps / denoise)))
    sig = get_schedule(total, round(width * height / 256))
    if not torch.is_tensor(sig):
        sig = torch.tensor(sig, dtype=torch.float32)
    return sig[-(steps + 1):]


def ksample(model, latent, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise,
            noise_mask=None, pixel_size=None, sigmas=None):
    """Như common_ksampler nhưng hỗ trợ scheduler 'flux2' (custom sigmas), noise_mask và sigmas truyền thẳng
    (sigmas != None -> bỏ qua scheduler/denoise, chạy đúng dãy sigmas đó)."""
    lat = comfy.sample.fix_empty_latent_channels(model, latent)
    noise = comfy.sample.prepare_noise(lat, seed, None)
    sched = scheduler
    if sigmas is not None:
        steps, sched, denoise = len(sigmas) - 1, "simple", 1.0
    elif scheduler == "flux2":
        w, h = pixel_size
        sigmas = flux2_sigmas(steps, w, h, denoise)
        sched = "simple"
    cb = latent_preview.prepare_callback(model, steps)
    return comfy.sample.sample(model, noise, steps, cfg, sampler_name, sched, positive, negative, lat,
                               denoise=denoise, noise_mask=noise_mask, sigmas=sigmas, callback=cb,
                               disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)


# ----------------------------------------------------------------------------
# Differential Diffusion (denoise theo vùng) — dùng code gốc ComfyUI nếu có
# ----------------------------------------------------------------------------
def _dd_forward(sigma, denoise_mask, extra_options):
    try:
        from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion
        return DifferentialDiffusion.forward(sigma, denoise_mask, extra_options, strength=1.0)
    except (ImportError, AttributeError, TypeError):
        ms = extra_options["model"].inner_model.model_sampling
        step_sigmas = extra_options["sigmas"]
        sigma_to = ms.sigma_min if step_sigmas[-1] <= ms.sigma_min else step_sigmas[-1]
        ts_from, ts_to = ms.timestep(step_sigmas[0]), ms.timestep(sigma_to)
        thr = (ms.timestep(sigma[0]) - ts_to) / (ts_from - ts_to)
        return (denoise_mask >= thr).to(denoise_mask.dtype)


def with_regional_denoise(model):
    m = model.clone()
    m.set_model_denoise_mask_function(_dd_forward)
    return m


# ----------------------------------------------------------------------------
# ControlNet adapters
# ----------------------------------------------------------------------------
def apply_native_cn(positive, negative, cn, vae, control_img, strength, start, end):
    """CONTROL_NET chuẩn ComfyUI (Flux, SDXL...). control_img [B,H,W,3] (B = số ô trong lượt)."""
    import nodes as comfy_nodes
    node = comfy_nodes.ControlNetApplyAdvanced()
    try:
        return node.apply_controlnet(positive, negative, cn, control_img, strength, start, end, vae=vae)
    except TypeError:  # bản ComfyUI cũ không có tham số vae
        return node.apply_controlnet(positive, negative, cn, control_img, strength, start, end)


def apply_qwen21_cn(model, cn, vae, control_img, strength, start, end):
    """Qwen-Image 2.1 Fun ControlNet (node b2renger) — patch model."""
    import nodes as comfy_nodes
    cls = comfy_nodes.NODE_CLASS_MAPPINGS.get("QwenImage21FunControlNetApply")
    if cls is None:
        raise RuntimeError("Chưa cài ComfyUI-QwenImage21-FunControlNet (QwenImage21FunControlNetApply)")
    return cls().apply(model, cn, vae, control_img, strength, start, end)[0]


def is_shape_mismatch(err):
    s = str(err).lower()
    return any(k in s for k in ("cannot be multiplied", "mat1 and mat2", "size mismatch"))
