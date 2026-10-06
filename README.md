# DB9_Ultimate — v0.12.0

Bộ node ComfyUI upscale theo ô 6K–11K cho mọi checkpoint (FLUX.2 Klein, Qwen-Image 2.1, model khác qua profile JSON).
Gộp phần tốt nhất của `db9_flux_locked_upscale` và `DB9_UpscaleEnchanceIMAGE`.

## Cài đặt
Clone/copy vào `ComfyUI/custom_nodes/DB9_Ultimate`, khởi động lại. Chỉ cần torch (+ opencv nếu có, cho canny).
Test (chạy bằng python của ComfyUI, từ thư mục DB9_Ultimate):
```
..\..\..\python_embeded\python.exe tests\test_core.py
..\..\..\python_embeded\python.exe tests\test_engine_mock.py
..\..\..\python_embeded\python.exe tests\test_forge_mock.py
```
Cả hai phải in `ALL PASS`.

## Nodes (`DB9 Ultimate`)
| Node | Việc |
|---|---|
| **DB9U Upscale** | Node chính. Chỉ chỉnh `target`, `denoise`, `cfg` (0 = auto), `preset`, `seed`. Nối thêm (tuỳ chọn): upscale_model, control_net (chuẩn), qwen21_controlnet, control_image, settings |
| **DB9U Forge** | Node riêng gộp 2 bộ: **tạo hình kiểu db9_flux_locked_upscale** (AI vẽ ô ở x2 rồi thu về, khoá màu chỉ chỉnh thống kê tần số thấp → giữ form AI, unsharp) + **ghép hình kiểu Ultimate** (ô đều bội 32, dốc tuyến tính đúng vùng giao, căn trôi, QA, flow_align, lô/VRAM, resume). Xem mục Forge bên dưới |
| **DB9U Advanced Settings** | Tinh chỉnh sâu (0 / -1 / auto = tự động) |
| **DB9U Finish** | 1 node: so sánh trước/sau + Editor kiểu Lightroom (Basic, Tone Curve, HSL, Color Grading, Selective, Effects, LUT). Chỉnh màu chạy trong trình duyệt, không chạy lại workflow; chỉ **Lưu full-res** mới chạy lệnh |
| **DB9U SR Refine** | Làm nét bằng upscale model (4x-UltraSharp/Nomos/DAT) rồi thu về ĐÚNG kích thước cũ (supersampling, chạy theo ô). `detail_only` (mặc định) chỉ lấy chi tiết mịn → không lệch màu/viền. Đặt sau DB9U Upscale, trước Finish. Chỉnh strength/mode/detail_sigma rồi Run: chỉ trộn lại (vài giây), không chạy lại upscale lẫn model — seed DB9U Upscale để `fixed` |
| **DB9U Save** | Lưu local_folder / output ComfyUI (cloud tự nhận), png16/tiff16/png8/jpg95; xuất cặp preview cho node **Compare Images** (thanh trượt trước/sau) |
| **Color (7 node)** | Basic · Presence (clarity/texture/dehaze/vignette/grain) · Curves · HSL · Color Balance · Selective Color · LUT .cube — nối tiếp IMAGE→IMAGE |
| **DB9U QAQC Compare** | ΔE, lệch dx/dy, méo cục bộ, SSIM, F1 viền, heatmap, auto_fix |

## Tự động
- **Profile model** (`db9u/profiles/*.json`): tự nhận Qwen-Image 2.1 / FLUX.2 → steps, cfg, sampler, scheduler (Flux2Scheduler), cỡ ô gốc, kiểu ControlNet. Model mới: thêm 1 file JSON.
- **Phần cứng**: <11GB → ô 1024 · <15GB → 1536 · ≥15GB → theo profile. ≥40GB chạy 2 ô/lô, ≥70GB 4 ô/lô (A100/H100). Hết VRAM → giảm lô rồi giảm ô, chạy lại.
- **Lượt**: tự chia ≤2x/lượt, lượt cuối khớp đúng kích thước, khoá màu cuối theo ảnh gốc.
- **Denoise theo vùng**: lá/cỏ/texture = denoise + 0.2 (≤0.85), kiến trúc = denoise (Differential Diffusion gốc ComfyUI).
- **Giữ form/màu**: căn ô trôi (sub-pixel), QA + chạy lại ô lệch, khoá màu Lab (chỉ tần số thấp), khoá cạnh kiến trúc (bỏ qua lá/texture).
- **ControlNet**: chuẩn ComfyUI (tự tắt khi không tương thích) hoặc Qwen 2.1 Fun; ảnh control tự tạo theo từng ô (tile/canny/lineart/grayscale) hoặc external.
- **LoRA**: nối qua MODEL/CLIP như bình thường.

## Tiện ích (Advanced Settings)
- `preview_tile` bật + `preview_tiles` trống → chỉ vẽ **bảng ô** O1..On (vài giây, không chạy model), 3 ô khung cam = nhiều chi tiết nhất.
- `preview_tiles` = `O1,O5,O6` · `O3-O6` · `auto` (ô khó nhất) · `auto3` → chỉ chạy các ô đó ở kích thước cuối, ghép thành bảng để so sánh/chỉnh denoise-cfg trước khi chạy cả ảnh.
- **Bấm chọn ô**: sau khi vẽ bảng ô, bấm nút `🔲 Chọn ô trên bảng` (trên DB9U Upscale hoặc Advanced Settings) → bấm ô để chọn/bỏ, "3 ô khó nhất", ô nhập so denoise → **Áp dụng** (điền `preview_tiles`) hoặc **Áp dụng & chạy**.
- `preview_layout` = `frame` (mặc định): ô preview được đặt đúng chỗ trên cả khung ảnh kích thước cuối (phần còn lại = ảnh phóng thường), `original_resized` = ảnh gốc cùng khung → so sánh trực tiếp trong Finish/QAQC. `grid` = ghép bảng các ô như cũ.
- `preview_denoise` = `0.3,0.4,0.5` → mỗi ô preview chạy ở từng mức, bảng hàng = ô, cột = denoise (zone map dịch theo mức).
- `reuse_preview` (bật): ảnh chỉ cần 1 lượt + `resume` bật → ô preview ở denoise chính được lưu; tắt `preview_tile` chạy full sẽ **dùng lại** ô đó (giữ nguyên ảnh/prompt/thông số khác). Ảnh >1 lượt thì không dùng lại được (log báo).
- `enhance` 0..1 (mặc định 1): 1 = AI đẩy chi tiết — giữ vân vừa + chi tiết mịn của AI, chỉ khoá màu/mảng lớn (> `lock_sigma`) theo ảnh gốc (cách repo db9_flux_locked_upscale). 0 = bám gốc (band fusion: vân vừa lấy từ SR, AI chỉ thêm chi tiết li ti qua cổng tương quan).
- `structure_lock` 0..1 (mặc định 0.5): khi enhance, đo tương quan cục bộ từng dải tần giữa AI và SR/ảnh gốc; chỗ AI vẽ lại hình (lá, cỏ, vân đá đổi dạng) thì dải đó lấy từ SR/gốc. 0 = nhận hết AI.
- `despeckle` 0..1 (mặc định 0): khử đốm/hạt nhiễu render ở mảng phẳng (tường vữa, trần) trên ảnh gốc TRƯỚC khi phóng. Lá/texture dày và cạnh kiến trúc không bị động. Cao → sạch hơn nhưng mất bớt vân vữa thật; thử 0.3–0.5.
- `resume` (mặc định bật): ô đã xong lưu ở `output/db9u_cache`; crash/ngắt → chạy lại cùng ảnh+thông số sẽ bỏ qua ô đã xong. Xong ảnh tự xoá cache.
- `sky_denoise`: trời tự nhận (vùng phẳng sáng nối mép trên) giữ gần nguyên (0.12).
- Output `zone_map`: sáng = denoise cao (lá), tối = thấp (trời).

## DB9U Forge (tạo hình Flux + ghép Ultimate)
Nối giống workflow flux cũ: Load Checkpoint → LoRA upscale → CLIP Text Encode (positive = negative được) + Load Upscale Model 4x-UltraSharp → **DB9U Forge** → Save/Finish. Mở sẵn: `workflows/DB9U_Forge_Flux2Klein9B.json`.

| | Bộ flux cũ | DB9U Upscale | **DB9U Forge** |
|---|---|---|---|
| AI vẽ ở | x2 ảnh ra (gián tiếp qua TileSplit + scale 2 + TileMerge) | đúng kích thước ra | x`supersample` (mặc định 2) |
| Khoá màu ô | mean/std tần số thấp, giữ form AI | thay hẳn tần số thấp bằng gốc (mất form AI) | như flux (pad reflect, không tối viền) |
| Chia ô | 2 tầng, ô cuối đè gần hết, giao 32 | ô đều bội 32 | như Ultimate, 1 tầng |
| Ghép | cosine feather 24 + chia trọng số | dốc tuyến tính = vùng giao, căn trôi, flow | như Ultimate |
| cfg khi pos = neg | chạy cfg 8 (tốn x2, kết quả = cfg 1) | — | tự ép cfg 1 |
| Hết VRAM / crash | lỗi | hạ lô/ô, resume | hạ lô/ô, resume |

Thông số gốc: supersample 2.0, color_lock 0.55, contrast_lock 0.2, sharpen 0.4–0.65, global_color 0.5, scheduler `simple` (hoặc auto = flux2), work_tile 2048 nếu VRAM ≥ 24GB.
- `denoise_mode` = **true** (mặc định, v0.12.1): `denoise` = đúng mức nhiễu bắt đầu (0.5 → 50% nhiễu). Kiểu `scheduler` cũ của Comfy với flux2 cắt đuôi lịch → denoise 0.5 thực tế nhiễu ~0.86 (gần vẽ lại hết → méo). Log in `denoise thật: nhiễu bắt đầu X`. Test thật (Klein, ref strip): **0.40–0.50 enhance sạch** (0.45 mặc định, ΔE ~2), 0.55+ bắt đầu vẽ lại chất liệu (mây thành lưới). ref tile = chỉ upscale (ΔE ~0.7). Mặc định sharpen 0.15, ss_source lanczos. Cuối lượt log `mức enhance: ΔE trung bình` tự gợi ý tăng/hạ.
- `flow_guard` (px, mặc định 4): flow chỉ kéo chỗ AI trôi nhẹ (≤ guard) về đúng hình gốc; chỗ AI lệch > 2×guard (vẽ lại khác hẳn) giữ nguyên nét AI thay vì kéo giãn → hết méo kiểu "ảnh bị bóp". 0 = kéo hết như cũ. Log `căn flow p95 … / % ảnh lệch`.
- **Preview** (v0.12.3, như DB9U Upscale): bật `preview_tile`, `preview_tiles` trống → bảng ô → bấm `🔲 Chọn ô trên bảng` trên node Forge, gõ `0.4,0.45,0.5` → bảng so denoise + log `mức enhance` từng mức. Tắt preview chạy full dùng lại ô ở denoise chính.

Muốn màu bám gốc hơn: `chroma_lock` 0.5–1. Muốn viền kiến trúc thẳng tuyệt đối: `edge_guard` 0.5. Nhanh hơn: `ss_source` = lanczos hoặc supersample 1.5.

## Colab
`notebooks/DB9U_Colab.ipynb`: cài ComfyUI + DB9_Ultimate (cần GitHub token) + tải model Qwen 2.1 / FLUX.2 Klein 9B (cần HF token) + link public.

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

## Chưa làm
Caption tự động · đo vết nối trong QAQC · preset JSON cho Color Grade.

## Chưa kiểm chứng
- Chưa chạy với model thật (sandbox phát triển không có torch). Cần chạy 2 file test + 1 ảnh thật.
- Chạy lô nhiều ô (A100/H100) chưa thử thực tế.
