# DB9_Ultimate — v0.1.0

Bộ node ComfyUI upscale theo ô 6K–11K cho mọi checkpoint (FLUX.2 Klein, Qwen-Image 2.1, model khác qua profile JSON).
Gộp phần tốt nhất của `db9_flux_locked_upscale` và `DB9_UpscaleEnchanceIMAGE`.

## Cài đặt
Clone/copy vào `ComfyUI/custom_nodes/DB9_Ultimate`, khởi động lại. Chỉ cần torch (+ opencv nếu có, cho canny).
Test (chạy bằng python của ComfyUI, từ thư mục DB9_Ultimate):
```
..\..\..\python_embeded\python.exe tests\test_core.py
..\..\..\python_embeded\python.exe tests\test_engine_mock.py
```
Cả hai phải in `ALL PASS`.

## Nodes (`DB9 Ultimate`)
| Node | Việc |
|---|---|
| **DB9U Upscale** | Node chính. Chỉ chỉnh `target`, `denoise`, `cfg` (0 = auto), `preset`, `seed`. Nối thêm (tuỳ chọn): upscale_model, control_net (chuẩn), qwen21_controlnet, control_image, settings |
| **DB9U Advanced Settings** | Tinh chỉnh sâu (0 / -1 / auto = tự động) |
| **DB9U QAQC Compare** | ΔE, lệch dx/dy, méo cục bộ, SSIM, F1 viền, heatmap, auto_fix |

## Tự động
- **Profile model** (`db9u/profiles/*.json`): tự nhận Qwen-Image 2.1 / FLUX.2 → steps, cfg, sampler, scheduler (Flux2Scheduler), cỡ ô gốc, kiểu ControlNet. Model mới: thêm 1 file JSON.
- **Phần cứng**: <11GB → ô 1024 · <15GB → 1536 · ≥15GB → theo profile. ≥40GB chạy 2 ô/lô, ≥70GB 4 ô/lô (A100/H100). Hết VRAM → giảm lô rồi giảm ô, chạy lại.
- **Lượt**: tự chia ≤2x/lượt, lượt cuối khớp đúng kích thước, khoá màu cuối theo ảnh gốc.
- **Denoise theo vùng**: lá/cỏ/texture = denoise + 0.2 (≤0.85), kiến trúc = denoise (Differential Diffusion gốc ComfyUI).
- **Giữ form/màu**: căn ô trôi (sub-pixel), QA + chạy lại ô lệch, khoá màu Lab (chỉ tần số thấp), khoá cạnh kiến trúc (bỏ qua lá/texture).
- **ControlNet**: chuẩn ComfyUI (tự tắt khi không tương thích) hoặc Qwen 2.1 Fun; ảnh control tự tạo theo từng ô (tile/canny/lineart/grayscale) hoặc external.
- **LoRA**: nối qua MODEL/CLIP như bình thường.

## Profile có sẵn
| Profile | Nguồn thông số |
|---|---|
| qwen_image_21 | docs.comfy.org: 25 steps, cfg 1, euler/simple, 2K |
| flux2_klein_distilled | Comfy-Org workflow_templates: Flux2Scheduler 4 steps, cfg 1, euler |
| flux2_klein_base | Comfy-Org workflow_templates: Flux2Scheduler 20 steps, cfg 5, euler (chọn tay) |
| generic | an toàn cho model khác |

FLUX.2 dev / Klein base / Klein distilled dùng chung class model → tự nhận mặc định Klein distilled, log có cảnh báo; model khác chọn tay trong Advanced Settings.

## Workflows
`workflows/DB9U_Qwen21.json`, `workflows/DB9U_Flux2Klein9B.json`.

## Lộ trình
P2 Save & Compare (local/cloud, slider trước/sau) · P3 Color Grade (Lightroom-like) · P4 preview 1 ô, resume, zone map (trời/kính/lá), caption.

## Chưa kiểm chứng
- Chưa chạy với model thật (sandbox phát triển không có torch). Cần chạy 2 file test + 1 ảnh thật.
- Chạy lô nhiều ô (A100/H100) chưa thử thực tế.
