"""DB9U Finish: chỉnh màu + so sánh trước/sau (thanh trượt ngay trên node) + lưu file — 1 node.

Chế độ làm việc:
- save_file = False: chỉ chỉnh trên ảnh preview (nhanh). live_edit = True -> kéo thanh là tự chạy lại
  (chỉ node này chạy lại nhờ cache ComfyUI; seed node Upscale phải để fixed).
- save_file = True: áp chỉnh màu lên ảnh full-res và lưu (local / output / cloud).
"""
import os

import torch

import folder_paths
import nodes as comfy_nodes

from . import core, grade
from .save import EXT, FORMATS, _next_path, is_cloud, write_image

CATEGORY = "DB9 Ultimate"
S = grade._s


def grade_params_spec():
    spec = {
        # Basic
        "exposure": S(0.0, -5.0, 5.0, 0.05), "contrast": S(), "highlights": S(), "shadows": S(),
        "whites": S(), "blacks": S(), "temperature": S(), "tint": S(), "vibrance": S(), "saturation": S(),
        # Presence
        "clarity": S(), "texture": S(), "dehaze": S(), "vignette": S(), "grain": S(0.0, 0.0, 100.0),
        # Curves
        "curve_master": ("STRING", {"default": "", "tooltip": "x,y (0-255) cách nhau khoảng trắng. Vd: 0,0 64,52 192,206 255,255"}),
        "curve_red": ("STRING", {"default": ""}), "curve_green": ("STRING", {"default": ""}),
        "curve_blue": ("STRING", {"default": ""}),
    }
    for n in grade.HUES:  # HSL
        spec[f"hsl_{n}_hue"] = S()
        spec[f"hsl_{n}_sat"] = S()
        spec[f"hsl_{n}_lum"] = S()
    for tone in ("shadows", "midtones", "highlights"):  # Color Balance
        for ax in ("cyan_red", "magenta_green", "yellow_blue"):
            spec[f"cb_{tone}_{ax}"] = S()
    spec.update({
        "sel_target": (grade.SEL_TARGETS,), "sel_cyan": S(), "sel_magenta": S(), "sel_yellow": S(), "sel_black": S(),
        "lut": (grade.list_luts(),), "lut_path": ("STRING", {"default": ""}), "lut_strength": S(1.0, 0.0, 1.0, 0.05),
    })
    return spec


def apply_all(x, p, seed=0):
    x = grade.basic(x, p["exposure"], p["contrast"], p["highlights"], p["shadows"], p["whites"], p["blacks"],
                    p["temperature"], p["tint"], p["vibrance"], p["saturation"])
    if any(p[k] for k in ("clarity", "texture", "dehaze", "vignette", "grain")):
        x = grade.presence(x, p["clarity"], p["texture"], p["dehaze"], p["vignette"], p["grain"], seed)
    if any(p[k].strip() for k in ("curve_master", "curve_red", "curve_green", "curve_blue")):
        x = grade.curves(x.clone(), p["curve_master"], p["curve_red"], p["curve_green"], p["curve_blue"])
    adj = {n: (p[f"hsl_{n}_hue"], p[f"hsl_{n}_sat"], p[f"hsl_{n}_lum"]) for n in grade.HUES}
    x = grade.hsl_mix(x, adj)
    cb = lambda t: (p[f"cb_{t}_cyan_red"], p[f"cb_{t}_magenta_green"], p[f"cb_{t}_yellow_blue"])
    if any(any(cb(t)) for t in ("shadows", "midtones", "highlights")):
        x = grade.color_balance(x, cb("shadows"), cb("midtones"), cb("highlights"), True)
    if any(p[k] for k in ("sel_cyan", "sel_magenta", "sel_yellow", "sel_black")):
        x = grade.selective_color(x, p["sel_target"], p["sel_cyan"], p["sel_magenta"], p["sel_yellow"], p["sel_black"], True)
    path = p["lut_path"].strip().strip('"') or (os.path.join(grade.LUT_DIR, p["lut"]) if p["lut"] != "none" else "")
    if path:
        x = grade.apply_cube(x, grade.load_cube(path), p["lut_strength"])
    return x.clamp(0, 1)


class DB9U_Finish:
    @classmethod
    def INPUT_TYPES(cls):
        req = {
            "image": ("IMAGE",),
            "save_file": ("BOOLEAN", {"default": False, "tooltip": "Tắt: chỉ chỉnh trên preview (nhanh). Bật: áp full-res + lưu file"}),
            "live_edit": ("BOOLEAN", {"default": True, "tooltip": "Kéo thanh chỉnh là tự chạy lại (khi save_file tắt). Node Upscale để seed fixed"}),
            "filename_prefix": ("STRING", {"default": "DB9U/img"}),
            "save_to": (["auto", "comfy_output", "local_folder"], {"default": "auto"}),
            "local_folder": ("STRING", {"default": "", "tooltip": r"Vd J:\Render\Upscale. Trống = output ComfyUI; cloud tự lưu output"}),
            "format": (FORMATS, {"default": "png16"}),
            "preview_max": ("INT", {"default": 2048, "min": 512, "max": 8192, "step": 256}),
        }
        req.update(grade_params_spec())
        return {"required": req, "optional": {"original": ("IMAGE", {"tooltip": "Nối original_resized từ DB9U Upscale (A = trước)"})}}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "saved_path")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def run(self, image, save_file, live_edit, filename_prefix, save_to, local_folder, format, preview_max,
            original=None, **gp):
        img = image[..., :3].float().cpu()
        h, w = img.shape[1:3]
        k = min(1.0, preview_max / max(h, w))
        pw, ph = max(1, round(w * k)), max(1, round(h * k))
        prev = core.resize(img, pw, ph, "area") if k < 1 else img
        prev_g = apply_all(prev, gp)
        if original is not None:
            org = original[:1, ..., :3].float().cpu()
            org_p = core.resize(org, pw, ph, "area") if org.shape[1:3] != (ph, pw) else org
        else:
            org_p = prev  # không nối original -> so sánh trước/sau chỉnh màu
        paths, out = [], prev_g
        if save_file:
            out = torch.cat([apply_all(img[i:i + 1], gp) for i in range(img.shape[0])], 0)
            use_local = save_to == "local_folder" or (save_to == "auto" and local_folder.strip() and not is_cloud())
            root = local_folder.strip() if use_local and local_folder.strip() else folder_paths.get_output_directory()
            for i in range(out.shape[0]):
                path, _ = _next_path(root, filename_prefix, EXT[format])
                path, used = write_image(path, out[i], format)
                paths.append(path)
                print(f"[DB9U] Đã lưu {used}: {path}")
        pv = comfy_nodes.PreviewImage()
        a = pv.save_images(org_p[:1], "db9u.cmp.a")["ui"]["images"]
        b = pv.save_images(prev_g[:1], "db9u.cmp.b")["ui"]["images"]
        raw = pv.save_images(prev[:1], "db9u.cmp.raw")["ui"]["images"]  # ảnh chưa chỉnh: Editor chỉnh màu trong trình duyệt
        info = f"{w}x{h} · {'ĐÃ LƯU: ' + paths[0] if paths else 'preview (chưa lưu — bật save_file để lưu full-res)'}"
        return {"ui": {"db9u_a": a, "db9u_b": b, "db9u_raw": raw, "db9u_info": [info]},
                "result": (out, "\n".join(paths))}


# ---------------------------------------------------------------- API cho Editor: đọc file .cube (chỉ .cube)
try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/db9u/lut")
    async def _db9u_lut(request):
        name = os.path.basename(request.query.get("name", ""))
        path = request.query.get("path", "").strip().strip('"')
        full = path if path else os.path.join(grade.LUT_DIR, name)
        if not full.lower().endswith(".cube") or not os.path.isfile(full):
            return web.Response(status=404, text="Không thấy file .cube")
        with open(full, "r", encoding="utf-8", errors="ignore") as f:
            return web.Response(text=f.read(), content_type="text/plain")
except Exception:  # chạy ngoài ComfyUI (test)
    pass
