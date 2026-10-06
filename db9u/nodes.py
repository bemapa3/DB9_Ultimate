"""DB9 Ultimate — nodes."""
import torch

import comfy.samplers

from . import core, engine, fluxmatch, profiles, region
from .qa import DB9U_QAQC
from .save import DB9U_Save
from .finish import DB9U_Finish
from .refine import DB9U_SRRefine
from .forge import DB9U_Forge
from .grade import GRADE_NODES

CATEGORY = "DB9 Ultimate"


class DB9U_Upscale:
    """Node chính: chỉ cần chọn đích, denoise, cfg. Mọi thứ khác tự động (profile model + VRAM)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "target": (list(engine.TARGETS.keys()), {"default": "8K",
                           "tooltip": "x2/x3/x4 = hệ số · 6K/8K/10K/11K = cạnh dài 6144/8192/10240/11264px. Tỉ lệ ảnh giữ nguyên"}),
                "denoise": ("FLOAT", {"default": 0.40, "min": 0.05, "max": 1.0, "step": 0.01,
                            "tooltip": "Kiến trúc/mảng phẳng. Vùng lá/texture tự +0.2. Khuyến nghị 0.30-0.55"}),
                "cfg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1,
                        "tooltip": "0 = tự theo model (Qwen 2.1 / Klein distilled: 1 · Klein base: 5)"}),
                "preset": (engine.PRESETS, {"default": "balanced",
                           "tooltip": "fast: ít bước, không chạy lại ô · balanced · max: thêm bước, giao rộng, chạy lại ô lệch tới 2 lần"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL", {"tooltip": "Phóng trước bằng model (chỉ lượt đầu). Không nối = lanczos"}),
                "control_net": ("CONTROL_NET", {"tooltip": "ControlNet chuẩn ComfyUI (Flux, SDXL...). Không tương thích -> tự tắt"}),
                "qwen21_controlnet": ("QWEN21_FUN_CONTROLNET", {"tooltip": "Từ node 'Load Qwen-Image 2.1 Fun ControlNet'"}),
                "control_image": ("IMAGE", {"tooltip": "Chỉ dùng khi control_mode = external (cả khung, cùng tỉ lệ)"}),
                "settings": ("DB9U_SETTINGS", {"tooltip": "Từ node DB9U Advanced Settings (không nối = tự động hết)"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("image", "original_resized", "log", "zone_map")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kw):
        # đổi sau mỗi lần bấm Reset preview -> ComfyUI chạy lại node (docs.comfy.org: IS_CHANGED)
        return engine.RESET_TOKEN[0]

    def run(self, image, model, vae, positive, negative, target, denoise, cfg, preset, seed,
            upscale_model=None, control_net=None, qwen21_controlnet=None, control_image=None, settings=None):
        job = engine.Job(model, denoise, cfg, preset, seed, settings, qwen21_controlnet, control_net)
        eng = engine.Engine(job, model, vae, positive, negative, control_image)
        eng.say(job.describe())
        outs, origs, masks = [], [], []
        for bi in range(image.shape[0]):
            img = image[bi:bi + 1, ..., :3].float().cpu()
            if job.s["preview_tile"]:
                out, orig, tex = eng.run_preview(img, target, upscale_model)
            else:
                out, tex = eng.run_image(img, target, upscale_model)
                orig = core.resize(img, out.shape[2], out.shape[1], "bicubic")
            outs.append(out)
            origs.append(orig)
            masks.append(tex)
        if len(outs) > 1:
            ref = outs[0].shape
            outs = [o if o.shape == ref else core.resize(o, ref[2], ref[1], "bicubic") for o in outs]
            origs = [o if o.shape == ref else core.resize(o, ref[2], ref[1], "bicubic") for o in origs]
            masks = [m if m.shape[1:3] == ref[1:3] else core.resize(m, ref[2], ref[1], "bilinear") for m in masks]
        result = (torch.cat(outs, 0), torch.cat(origs, 0), "\n".join(eng.log), torch.cat(masks, 0))
        ui = {}
        if eng.board is not None:  # bảng ô -> JS cho bấm chọn ô
            try:
                import nodes as comfy_nodes
                im = comfy_nodes.PreviewImage().save_images(outs[0][:1], "db9u.board")["ui"]["images"]
                ui["db9u_board"] = [dict(eng.board, image=im[0])]
            except Exception as e:
                print(f"[DB9U] không gửi được bảng ô lên giao diện: {e}")
        return {"ui": ui, "result": result}


class DB9U_Settings:
    """Tuỳ chỉnh sâu (không bắt buộc). Giá trị 0 / -1 / auto = để node tự quyết."""

    @classmethod
    def INPUT_TYPES(cls):
        d = engine.DEFAULTS
        samplers = ["auto"] + list(comfy.samplers.KSampler.SAMPLERS)
        schedulers = ["auto", "flux2"] + list(comfy.samplers.KSampler.SCHEDULERS)
        return {"required": {
            "profile": (profiles.PROFILE_CHOICES, {"default": "auto", "tooltip": "auto = tự nhận model. Klein base cần chọn tay"}),
            "custom_long_edge": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 16, "tooltip": ">0: bỏ qua target"}),
            "max_tile": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32, "tooltip": "0 = auto (profile + VRAM)"}),
            "overlap": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 32, "tooltip": "0 = auto"}),
            "max_scale_per_pass": ("FLOAT", {"default": d["max_scale_per_pass"], "min": 1.2, "max": 8.0, "step": 0.1}),
            "batch_tiles": ("INT", {"default": 0, "min": 0, "max": 16, "tooltip": "0 = auto theo VRAM"}),
            "steps": ("INT", {"default": 0, "min": 0, "max": 200, "tooltip": "0 = theo profile"}),
            "sampler": (samplers, {"default": "auto"}),
            "scheduler": (schedulers, {"default": "auto", "tooltip": "flux2 = như node Flux2Scheduler"}),
            "texture_denoise": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01,
                                "tooltip": "-1 = auto (denoise+0.2) · 0 = tắt denoise theo vùng"}),
            "sky_denoise": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01,
                            "tooltip": "-1 = auto 0.12 (trời gần như giữ nguyên) · ≥ denoise = tắt nhận trời"}),
            "min_denoise": ("FLOAT", {"default": d["min_denoise"], "min": 0.0, "max": 1.0, "step": 0.01}),
            "tile_retries": ("INT", {"default": -1, "min": -1, "max": 5, "tooltip": "-1 = theo preset"}),
            "tile_max_deltaE": ("FLOAT", {"default": d["tile_max_deltaE"], "min": 0.5, "max": 50.0, "step": 0.1}),
            "align_tiles": ("BOOLEAN", {"default": True}),
            "seed_mode": (["fixed", "per_tile"], {"default": "fixed"}),
            "reference_mode": (["auto", "strip", "tile", "keep"], {"default": "auto"}),
            "color_lock": (core_color_modes(), {"default": d["color_lock"]}),
            "lock_strength": ("FLOAT", {"default": d["lock_strength"], "min": 0.0, "max": 1.0, "step": 0.05}),
            "lock_sigma": ("FLOAT", {"default": d["lock_sigma"], "min": 0.5, "max": 64.0, "step": 0.5}),
            "edge_guard": ("FLOAT", {"default": d["edge_guard"], "min": 0.0, "max": 1.0, "step": 0.05}),
            "control_mode": (core.CONTROL_MODES, {"default": "auto"}),
            "control_strength": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 2.0, "step": 0.05, "tooltip": "-1 = auto"}),
            "control_start": ("FLOAT", {"default": d["control_start"], "min": 0.0, "max": 1.0, "step": 0.01}),
            "control_end": ("FLOAT", {"default": d["control_end"], "min": 0.0, "max": 1.0, "step": 0.01}),
            "resume": ("BOOLEAN", {"default": True, "tooltip": "Lưu ô đã xong (output/db9u_cache); chạy lại cùng ảnh+thông số thì dùng tiếp"}),
            "preview_tile": ("BOOLEAN", {"default": False, "tooltip": "Bật: chế độ xem trước. preview_tiles trống = chỉ vẽ bảng ô O1..On (vài giây)"}),
            "texture_lock_mult": ("FLOAT", {"default": d["texture_lock_mult"], "min": 1.0, "max": 16.0, "step": 0.5,
                                  "tooltip": "Vùng lá/texture: khoá màu chỉ ở mảng lớn (lock_sigma x số này). 1 = khoá như vùng khác (dễ bóng ma, nhoè lá)"}),
            "material_denoise": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01,
                                 "tooltip": "Vùng tường/đá/gỗ/mảng phẳng. -1 = auto min(denoise, 0.4). Thấp -> ít vân AI bịa"}),
            "ai_detail": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05,
                          "tooltip": "Cần nối upscale_model. Mức chi tiết mịn AI được thêm; vân vừa luôn lấy từ SR ảnh gốc. 0 = chỉ SR, 1 = AI tối đa"}),
            "detail_sigma": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 8.0, "step": 0.25,
                             "tooltip": "Ranh giới 'chi tiết mịn' (px). Nhỏ -> AI chỉ thêm chi tiết li ti, bám gốc hơn"}),
            "fidelity_gate": ("BOOLEAN", {"default": True,
                              "tooltip": "Tự bỏ vân AI không khớp vân gốc (đốm/hạt trên tường)"}),
            "flow_align": ("BOOLEAN", {"default": True, "tooltip": "Căn AI về đúng vị trí gốc từng pixel trước khi trộn (hết bóng mờ). Cần OpenCV"}),
            "texture_ai": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                           "tooltip": "Vùng lá/texture: pha thêm AI trọn. 0 = bám gốc nhất. Tăng nếu muốn lá 'vẽ lại' nhiều hơn"}),
            "preview_tiles": ("STRING", {"default": "", "tooltip": "Khi preview_tile bật: trống = bảng ô · O1,O5,O6 · O3-O6 · auto (ô khó nhất) · auto3. Hoặc bấm '🔲 Chọn ô trên bảng'"}),
            "preview_denoise": ("STRING", {"default": "", "tooltip": "So nhiều denoise trên ô preview, vd 0.3,0.4,0.5 (hàng = ô, cột = denoise). Trống = chỉ denoise chính"}),
            "reuse_preview": ("BOOLEAN", {"default": True, "tooltip": "Chạy full (ảnh chỉ 1 lượt, resume bật) dùng lại ô đã preview ở denoise chính -> không chạy model lại ô đó"}),
            "despeckle": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                          "tooltip": "Khử đốm/hạt nhiễu render ở mảng phẳng (tường vữa, trần) TRƯỚC khi phóng. 0 = tắt. Cao -> sạch hơn nhưng mất bớt vân vữa thật. Lá/cạnh không bị động"}),
            "preview_layout": (["frame", "grid"], {"default": "frame",
                               "tooltip": "frame: ô preview đặt đúng chỗ trên cả khung ảnh kích thước cuối -> so sánh trực tiếp với ảnh gốc · grid: ghép bảng các ô (so nhiều denoise luôn dùng grid)"}),
            "enhance": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                        "tooltip": "1 = AI đẩy chi tiết tối đa (giữ vân + chi tiết AI, chỉ khoá màu/mảng lớn theo gốc) · 0 = bám gốc (vân vừa lấy từ SR, AI chỉ thêm chi tiết li ti) · 0.5 = giữa"}),
            "structure_lock": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                               "tooltip": "Chống biến dạng khi enhance: chỗ AI vẽ lại hình (lá, cỏ, vân đá đổi dạng) thì dùng chi tiết SR thay. 0 = nhận hết AI · 0.5 cân bằng · 1 = chặt (ít bịa, ít enhance)"}),
            "async_qa": ("BOOLEAN", {"default": True,
                         "tooltip": "QA + lưu cache ô chạy song song lúc GPU sample ô kế -> GPU ít phải chờ CPU. Tắt = như bản cũ (tuần tự)"}),
            "preview_pick": (list(engine.PREVIEW_PICKS), {"default": "ô đã chọn",
                             "tooltip": "Chọn ô preview bằng menu (tự bật preview_tile). 'ô đã chọn' = ô bấm trên bảng (🔲 Chọn ô trên bảng); chưa chọn ô nào = vẽ bảng ô"}),
            "chroma_lock": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                            "tooltip": "Màu ở chi tiết lấy từ ảnh gốc (SR), AI chỉ góp độ sáng (nếp, vân, sợi). Chặn cành tím / lá hồng / vân đá xanh đỏ. 0 = như cũ"}),
        }}

    RETURN_TYPES = ("DB9U_SETTINGS",)
    RETURN_NAMES = ("settings",)
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, **kw):
        return (dict(kw),)


class DB9U_FluxMatch:
    """1 thanh trượt kéo DB9U về kiểu bộ node flux cũ (db9_flux_locked_upscale). Nối: Advanced Settings -> node này -> settings."""

    @classmethod
    def INPUT_TYPES(cls):
        tip = "\n".join(f"{lv[0]:.2f} = {lv[1]} | Ưu: {lv[2]} | Nhược: {lv[3]}" for lv in fluxmatch.LEVELS)
        return {"required": {
            "flux_like": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": tip}),
            "keep_color": ("BOOLEAN", {"default": True,
                           "tooltip": "BẬT: lấy chi tiết kiểu flux nhưng màu vẫn khoá theo ảnh gốc (chroma_lock). "
                                      "TẮT: màu trôi theo AI như bộ flux cũ (ΔE cục bộ p95 ~13) — chỉ tắt khi muốn giống hệt"}),
        }, "optional": {"settings": ("DB9U_SETTINGS",)}}

    RETURN_TYPES = ("DB9U_SETTINGS", "STRING")
    RETURN_NAMES = ("settings", "note")
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, flux_like, keep_color, settings=None):
        s, note = fluxmatch.apply(settings, flux_like, keep_color)
        print(note)
        return (s, note)


REGION_TIP = ("Tô vùng chưa ưng bằng MaskEditor (chuột phải Load Image -> Open in MaskEditor). "
              "Tô trên ảnh thu nhỏ cũng được, node tự phóng mask. Mỗi mảng tô rời = 1 vùng chạy riêng")


class DB9U_RegionFix:
    """Tất cả trong 1: tô vùng chưa ok -> chạy lại riêng vùng đó với thông số khác -> ghép lại đúng từng pixel."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Ảnh đã upscale (ảnh cần sửa). Ghép lại đúng ảnh này"}),
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING", {"tooltip": "Có thể dùng prompt riêng cho vùng (vd chỉ tả vật liệu vùng đó)"}),
                "negative": ("CONDITIONING",),
                "denoise": ("FLOAT", {"default": 0.35, "min": 0.05, "max": 1.0, "step": 0.01}),
                "cfg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1, "tooltip": "0 = tự theo model"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "padding": ("INT", {"default": 96, "min": 0, "max": 512, "step": 16,
                            "tooltip": "Nới thêm ngữ cảnh quanh vùng cho model nhìn (không bị dán đè)"}),
                "feather": ("INT", {"default": 24, "min": 0, "max": 256, "step": 2, "tooltip": "Độ mềm mép hoà (px)"}),
                "work_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.25,
                               "tooltip": ">1: phóng vùng lên để model vẽ chi tiết hơn rồi thu về đúng cỡ (vùng nhỏ nên 1.5-2)"}),
                "color_match": ("BOOLEAN", {"default": True, "tooltip": "Khoá màu/mảng lớn theo vùng cũ, chỉ nhận chi tiết mới"}),
            },
            "optional": {
                "region_mask": ("MASK", {"tooltip": REGION_TIP}),
                "region_box": ("STRING", {"default": "", "tooltip": "Hoặc gõ toạ độ px: x,y,w,h ; x,y,w,h"}),
                "control_net": ("CONTROL_NET",),
                "qwen21_controlnet": ("QWEN21_FUN_CONTROLNET",),
                "settings": ("DB9U_SETTINGS", {"tooltip": "Dùng chung Advanced Settings (enhance, structure_lock, sampler...)"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("image", "before_after", "log", "region_mask")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, model, vae, positive, negative, denoise, cfg, seed, padding, feather, work_scale,
            color_match, region_mask=None, region_box="", control_net=None, qwen21_controlnet=None, settings=None):
        s = dict(settings or {})
        s["resume"], s["preview_tile"] = False, False
        job = engine.Job(model, denoise, cfg, "balanced", seed, s, qwen21_controlnet, control_net)
        eng = engine.Engine(job, model, vae, positive, negative)
        eng.say(job.describe())
        out = image[:1, ..., :3].float().cpu()
        infos = region.make_regions(out, region_mask, region_box, padding, feather)
        eng.say(f"[DB9U] Region Fix: {len(infos)} vùng | padding {padding} feather {feather} work_scale {work_scale}")
        full = torch.zeros(1, out.shape[1], out.shape[2])
        pairs = []
        for k, info in enumerate(infos):
            c = region.crop(out, info)
            h, w = c.shape[1], c.shape[2]
            if work_scale > 1.001:
                w2 = max(core.ALIGN, int(round(w * work_scale / core.ALIGN)) * core.ALIGN)
                h2 = max(core.ALIGN, int(round(h * work_scale / core.ALIGN)) * core.ALIGN)
                src = engine.lanczos(c, w2, h2)
            else:
                src = c
            merged, _ = eng.run_pass(src, f"vùng {k + 1}/{len(infos)}")
            fixed = eng.finish_pass(merged, src, None)
            out, msg = region.stitch(out, fixed, info, color_match, 1.0, job.s["lock_sigma"])
            eng.say("[DB9U]   " + msg)
            pairs.append((c, region.crop(out, info)))
            x0, y0, x1, y1 = info["crop"]
            full[:, y0:y1, x0:x1] = torch.maximum(full[:, y0:y1, x0:x1], info["mask"])
        return (out, region.before_after(pairs), "\n".join(eng.log), full)


class DB9U_RegionSource:
    """Nạp ảnh LỚN (6K-11K, PNG/TIFF 16-bit) thẳng từ đĩa — không kéo thả/upload (tránh lỗi 'file quá lớn').
    Đồng thời tạo ảnh proxy nhỏ 'db9u_mask_<tên>.jpg' trong thư mục input để tô mask ở Load Image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "path": ("STRING", {"default": "", "tooltip": "Đường dẫn đầy đủ (J:\\...\\img_00012.png) hoặc đường dẫn trong output ComfyUI (DB9U/img_00012.png). "
                                "Bỏ trống nếu nối image"}),
            "proxy_max": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 256,
                          "tooltip": "Cạnh dài ảnh proxy để tô mask"}),
        }, "optional": {
            "image": ("IMAGE", {"tooltip": "Nối thẳng output DB9U Upscale (không cần path)"}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "proxy_name")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True  # chạy được một mình để tạo proxy trước khi tô mask

    def run(self, path, proxy_max, image=None):
        import os
        import folder_paths
        if image is not None:
            img, name = image[:1, ..., :3].float().cpu(), "upscale"
        else:
            p = path.strip().strip('"')
            if not p:
                raise ValueError("Nhập path ảnh lớn hoặc nối image")
            if not os.path.isabs(p):
                p = os.path.join(folder_paths.get_output_directory(), p)
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Không thấy file: {p}")
            img, name = region.load_image_file(p), os.path.splitext(os.path.basename(p))[0]
        proxy = f"db9u_mask_{name}.jpg"
        pw, ph = region.write_proxy(img, os.path.join(folder_paths.get_input_directory(), proxy), proxy_max)
        print(f"[DB9U] Region Source: {img.shape[2]}x{img.shape[1]} | proxy {pw}x{ph} -> input/{proxy} "
              f"(ở Load Image bấm R để làm mới danh sách, chọn file này rồi Open in MaskEditor)")
        return {"ui": {"text": [proxy]}, "result": (img, proxy)}


class DB9U_RegionCrop:
    """Cắt vùng (kèm REGION_INFO) để tự chạy KSampler / inpaint riêng, rồi ghép bằng DB9U Region Stitch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "padding": ("INT", {"default": 96, "min": 0, "max": 512, "step": 16}),
            "feather": ("INT", {"default": 24, "min": 0, "max": 256, "step": 2}),
            "work_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.25}),
        }, "optional": {
            "region_mask": ("MASK", {"tooltip": REGION_TIP + " (nhiều mảng -> gộp thành 1 vùng)"}),
            "region_box": ("STRING", {"default": "", "tooltip": "x,y,w,h"}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "DB9U_REGION")
    RETURN_NAMES = ("crop", "crop_mask", "region_info")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, padding, feather, work_scale, region_mask=None, region_box=""):
        img = image[:1, ..., :3].float().cpu()
        info = region.make_regions(img, region_mask, region_box, padding, feather, merge_all=True)[0]
        c = region.crop(img, info)
        m = info["mask"]
        if work_scale > 1.001:
            w2 = max(core.ALIGN, int(round(c.shape[2] * work_scale / core.ALIGN)) * core.ALIGN)
            h2 = max(core.ALIGN, int(round(c.shape[1] * work_scale / core.ALIGN)) * core.ALIGN)
            c = engine.lanczos(c, w2, h2)
            m = torch.nn.functional.interpolate(m[:, None], size=(h2, w2), mode="bilinear", align_corners=False)[:, 0]
        return (c, m, info)


class DB9U_RegionStitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE", {"tooltip": "Đúng ảnh đã đưa vào Region Crop"}),
            "patch": ("IMAGE", {"tooltip": "Vùng đã chạy lại (cỡ khác sẽ tự đưa về đúng cỡ crop)"}),
            "region_info": ("DB9U_REGION",),
            "color_match": ("BOOLEAN", {"default": True}),
            "color_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("image", "before_after", "log")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, patch, region_info, color_match, color_strength):
        base = image[:1, ..., :3].float().cpu()
        out, msg = region.stitch(base, patch, region_info, color_match, color_strength)
        return (out, region.before_after([(region.crop(base, region_info), region.crop(out, region_info))]),
                "[DB9U] Region Stitch: " + msg)


def core_color_modes():
    return ["detail_transfer", "lab_stats", "both", "none"]


NODE_CLASS_MAPPINGS = {
    "DB9U_Upscale": DB9U_Upscale,
    "DB9U_Forge": DB9U_Forge,
    "DB9U_Settings": DB9U_Settings,
    "DB9U_FluxMatch": DB9U_FluxMatch,
    "DB9U_QAQC": DB9U_QAQC,
    "DB9U_Save": DB9U_Save,
    "DB9U_Finish": DB9U_Finish,
    "DB9U_SRRefine": DB9U_SRRefine,
    "DB9U_RegionFix": DB9U_RegionFix,
    "DB9U_RegionSource": DB9U_RegionSource,
    "DB9U_RegionCrop": DB9U_RegionCrop,
    "DB9U_RegionStitch": DB9U_RegionStitch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DB9U_Upscale": "DB9U Upscale (Ultimate)",
    "DB9U_Forge": "DB9U Forge (tạo hình Flux + ghép Ultimate)",
    "DB9U_Settings": "DB9U Advanced Settings",
    "DB9U_FluxMatch": "DB9U Flux Match (kéo về kiểu bộ flux cũ)",
    "DB9U_QAQC": "DB9U QAQC Compare",
    "DB9U_Save": "DB9U Save (local / cloud)",
    "DB9U_Finish": "DB9U Finish (Chỉnh màu + So sánh + Lưu)",
    "DB9U_SRRefine": "DB9U SR Refine (làm nét, giữ kích thước)",
    "DB9U_RegionFix": "DB9U Region Fix (sửa vùng + ghép chuẩn)",
    "DB9U_RegionSource": "DB9U Region Source (nạp ảnh lớn + proxy tô mask)",
    "DB9U_RegionCrop": "DB9U Region Crop",
    "DB9U_RegionStitch": "DB9U Region Stitch",
}

for _k, (_c, _n) in GRADE_NODES.items():
    NODE_CLASS_MAPPINGS[_k] = _c
    NODE_DISPLAY_NAME_MAPPINGS[_k] = _n


# ---------------------------------------------------------------- API nút "Reset preview" (web/js/db9u_tiles.js)
try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.post("/db9u/reset_preview")
    async def _db9u_reset_preview(request):
        try:
            n = engine.reset_preview_cache()
            print(f"[DB9U] Reset preview: đã xoá {n} bộ cache trong output/db9u_cache")
            return web.json_response({"ok": True, "removed": n, "token": engine.RESET_TOKEN[0]})
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)
except Exception:  # chạy ngoài ComfyUI (test)
    pass
