# Sổ tay thông số — DB9_Ultimate (upscale / enhance)

Ký hiệu: ✅ = đã test thật (Klein 9B, img_00027, v0.12.4) · ⚠️ = suy từ code/cơ chế, CHƯA test số liệu.
Cách đọc log: `mức enhance: ΔE trung bình` — <1.5 = chỉ upscale · 1.5–6 = enhance tốt · >6 = vẽ lại mạnh, dễ méo.

---

## 1. Muốn AI vẽ thêm chi tiết lá, vải, cây cối

### A. Dùng DB9U Forge (đang là bộ chính)
Forge dùng **1 mức denoise cho cả ảnh** (không chia vùng) → tăng chi tiết lá thì trời/tường cũng bị động.

| Thông số | Mặc định | Đẩy chi tiết | Ghi chú |
|---|---|---|---|
| `denoise` | 0.45 ✅ | **0.50 → 0.55** | Cần chỉnh nhất. 0.55+ trời/mây bắt đầu thành lưới ✅. Thử bằng preview trước |
| `supersample` | 2.0 ✅ | giữ 2.0 | AI vẽ ở x2 rồi thu về = nguồn chi tiết chính. Đừng hạ |
| `sharpen` | 0.15 ✅ | 0.25–0.30 ⚠️ | 0.4+ dễ gắt/sạn ✅ |
| `contrast_lock` | 0.20 | 0.10 ⚠️ | Thấp → AI giữ tương phản vi mô (gân lá, sợi vải) nhiều hơn |
| `color_lock` | 0.55 | giữ | Chỉ khoá màu mảng lớn, ít ảnh hưởng chi tiết |
| `chroma_lock` | 0 | **0.3–0.5** ⚠️ | Denoise cao dễ ra lá hồng/cành tím → bật để màu bám gốc, AI chỉ góp sáng tối |
| `flow_guard` | 4 | 3 ⚠️ | Nhỏ hơn → chỗ AI vẽ lại lá được giữ nguyên nét AI, không bị kéo giãn về hình cũ |
| `edge_guard` | 0.5 ✅ | giữ 0.5 | Giữ viền kiến trúc thẳng khi denoise cao |
| `reference_mode` | strip ✅ | strip | `tile` = bám chặt, gần như chỉ upscale (ΔE ~0.7) ✅ |
| `downscale` | bicubic_sharp | giữ | `area` mềm hơn nếu bị răng cưa |

**Quy trình gợi ý:**
1. Bật `preview_tile`, `preview_tiles` = `auto3` (3 ô nhiều chi tiết nhất — thường là cây).
2. `preview_denoise` = `0.45,0.5,0.55` → so bảng + đọc ΔE từng mức.
3. Chọn mức lá đẹp mà trời/tường chưa lưới. Nếu lá đổi màu → `chroma_lock` 0.4.
4. Tắt preview, chạy full (dùng lại ô đã preview).

### B. Dùng DB9U Upscale (có denoise theo vùng — hợp hơn khi ảnh nhiều cây + nhiều trời)
Node này tự nhận vùng lá/texture và cho denoise cao hơn ở đó, trời gần như giữ nguyên.

| Thông số (node / Advanced Settings) | Mặc định | Đẩy chi tiết lá/vải |
|---|---|---|
| `denoise` (kiến trúc) | 0.40 | giữ 0.35–0.45 |
| `texture_denoise` | -1 (= denoise + 0.2, tối đa 0.85) | 0.65–0.75 ⚠️ đặt tay |
| `texture_ai` | 0 | **0.3–0.6** ⚠️ — pha AI trọn vào vùng lá (0 = bám gốc nhất) |
| `structure_lock` | 0.5 | **0.2–0.3** ⚠️ — 0 = nhận hết hình AI vẽ lại (lá đổi dạng) |
| `enhance` | 1.0 | giữ 1.0 |
| `ai_detail` | 0.6 | 0.8 ⚠️ (cần nối upscale_model) |
| `texture_lock_mult` | mặc định | tăng nhẹ ⚠️ — khoá màu lá chỉ ở mảng rất lớn |
| `sky_denoise` | -1 (auto 0.12) | giữ — trời không bị vẽ lại |
| `material_denoise` | -1 (min(denoise,0.4)) | giữ — tường/gỗ ít vân bịa |

Xem output `zone_map`: sáng = vùng được denoise cao. Nếu vải/rèm không sáng → node chưa coi là texture, dùng Forge.

### C. Prompt (⚠️ chưa test)
Positive có thể thêm: `highly detailed foliage, sharp leaf veins, fabric weave texture, natural bark`.
Klein distilled chạy cfg 1 theo profile → chỉ positive có tác dụng, không chậm thêm. Nếu tự đặt cfg > 1 thì chạy chậm gấp đôi.

---

## 2. Chú thích từng thông số DB9U Forge

| Thông số | Mặc định | Làm gì | Tăng → | Giảm → |
|---|---|---|---|---|
| `target` | 6K | Cạnh dài đích (x2/x3/x4/6K/8K/10K/11K) | — | — |
| `denoise` | 0.45 | % nhiễu lúc bắt đầu (`denoise_mode` true) | AI vẽ thêm/vẽ lại nhiều | Gần như chỉ phóng to |
| `denoise_mode` | true | true = số đúng bằng % nhiễu · scheduler = kiểu KSampler (0.5 ≈ 86% nhiễu) | — | — |
| `supersample` | 2.0 | AI vẽ ở x lần ảnh ra rồi thu về | Chi tiết hơn, chậm hơn | 1.0 = vẽ thẳng, nhanh |
| `steps` / `cfg` / `sampler` / `scheduler` | 0 / 0 / auto / auto | 0/auto = theo profile model (Klein: 4 steps, cfg 1, flux2) | — | — |
| `seed_mode` | per_tile | Mỗi ô seed khác | — | — |
| `color_lock` | 0.55 | Khoá màu từng ô (mean/std tần số thấp) | Màu ô bám gốc | Ô lệch màu nhau |
| `contrast_lock` | 0.20 | Khoá tương phản từng ô | Bám gốc, phẳng hơn | AI giữ tương phản vi mô |
| `sharpen` | 0.15 | Unsharp trước khi thu về | Nét, dễ sạn | Mềm |
| `global_color` | 0.50 | Khớp màu cả ảnh theo gốc | Bám gốc | Giữ màu AI |
| `chroma_lock` | 0 | Màu chi tiết lấy từ gốc, AI chỉ góp sáng | Chặn lá hồng/cành tím | Màu AI tự do |
| `edge_guard` | 0.5 (chốt) | Khoá viền kiến trúc thẳng theo nền | Viền thẳng tuyệt đối | Giữ hình AI |
| `ss_source` | lanczos | Nguồn ô x supersample | upscale_model: như bộ flux, dễ sạn | — |
| `downscale` | bicubic_sharp | Cách thu ô về | — | area = mềm |
| `reference_mode` | strip | strip = img2img thuần · tile = gắn latent ô gốc | — | — |
| `work_tile` / `overlap` / `batch_tiles` | 0 | Cỡ ô / vùng giao / số ô 1 lượt (0 = auto VRAM) | Ít ô, cần VRAM | — |
| `align_tiles` | bật | Căn trôi sub-pixel từng ô | — | — |
| `flow_align` | bật | Căn AI về vị trí nền từng pixel (cần OpenCV) | — | — |
| `flow_guard` | 4 px | Chỉ kéo chỗ AI trôi < số px này | Kéo nhiều về hình gốc | Giữ nét AI vẽ lại |
| `tile_retries` / `tile_max_deltaE` | 0 / 8 | Ô lệch màu > ΔE thì chạy lại denoise thấp hơn | — | — |
| `resume` | bật | Lưu ô xong vào `output/db9u_cache` | — | — |
| `preview_*` | — | Xem trước ô / so nhiều denoise (xem mục 1A) | — | — |
| `upscale_model` | (nối) | 4x-UltraSharp làm nền | — | — |

## 3. Bộ thông số mẫu (Forge, Klein 9B)

| Mục đích | denoise | sharpen | chroma_lock | edge_guard | Ghi chú |
|---|---|---|---|---|---|
| Chỉ phóng to, bám gốc | 0.30 hoặc ref `tile` | 0.10 | 0.5 | 0.5 | ΔE < 1.5 |
| Enhance sạch (mặc định) ✅ | 0.45 | 0.15 | 0 | 0.5 | ΔE ~1.9, 60 ô 11.2 phút ở 6K |
| Đẩy chi tiết lá/vải ⚠️ | 0.50–0.55 | 0.25 | 0.4 | 0.5 | contrast_lock 0.10, flow_guard 3 |
| Vẽ lại mạnh (concept) ⚠️ | 0.60+ | 0.15 | 0.5 | 0.5 | Trời/mây dễ lưới, dễ méo |
