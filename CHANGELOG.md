# Changelog

## v0.12.3 — 2026-10-06
### ✅ Đã làm
- **Forge preview** (như DB9U Upscale, để test denoise nhanh): widget mới ở CUỐI node `preview_tile`, `preview_tiles` (trống = bảng ô · O1,O5 · O3-O6 · auto · auto3), `preview_denoise` (vd 0.7,0.8,0.85 -> bảng hàng = ô, cột = denoise), `preview_layout` (frame/grid).
- Mỗi mức denoise preview log `mức enhance (denoise X): ΔE trung bình` + gợi ý. Frame chạy cả hậu kỳ (flow có chặn, màu) nên log `căn flow` thấy luôn méo hay không.
- Ô preview ở denoise chính lưu cache (resume bật, khoá cache bỏ qua widget preview) -> tắt preview_tile chạy full dùng lại.
- Nút `🔲 Chọn ô trên bảng` có trên node Forge (điền thẳng vào widget Forge, không cần Settings).
- Tách `run_image` -> `header` / `make_base` / `post` dùng chung cho preview. Test: test_preview.

## v0.12.2 — 2026-10-06
### ✅ Đã làm
- Chạy thật v0.12.1 (denoise true 0.6, ref tile, simple): hết méo nhưng ΔE chỉ 0.3–1.1, flow p95 0.14px -> AI chép lại nền, **chỉ upscale không enhance**. Flux ở ô 1472px cần nhiễu cao mới đổi chi tiết (lịch flux2 tự đẩy lên ~0.86); ref tile còn neo thêm vào ô gốc.
- Forge denoise mặc định **0.6 -> 0.8** (workflow Forge cũng 0.8). Tooltip: ≤0.6 chép nền, 0.75–0.85 enhance, ≥0.86 vẽ lại/méo.
- Log cuối lượt `mức enhance: ΔE trung bình X` + gợi ý tự động (< 1.5 tăng denoise/ref strip, 1.5–6 tốt, > 6 hạ).

## v0.12.1 — 2026-10-06
### ✅ Đã làm
- Chạy thật lần đầu (5070 Ti, 4000x2250 -> 6K, Klein distilled, denoise 0.5, flux2): chất lượng ok nhưng méo hình. Log: căn flow `AI lệch p95 12.37px` (DB9U Upscale ~0.8px), 25/60 ô ΔE 8–14.
- Nguyên nhân chính: scheduler flux2 dịch nhiễu theo cỡ ô -> ở ô 1472x1408 denoise 0.3 = nhiễu bắt đầu 0.73, **0.5 = 0.86** (tính bằng get_schedule của ComfyUI) -> AI gần như vẽ lại hình. Phụ: căn flow kéo giãn ảnh theo hình gốc ở chỗ AI đã vẽ khác -> sinh méo.
- **Forge `denoise_mode`** (mặc định `true`): số denoise = đúng mức nhiễu lúc bắt đầu (giữ dáng đường cong của scheduler từ đó về 0, chạy bằng sigmas riêng). `scheduler` = như cũ. Log in mức nhiễu thật cả 2 chế độ. Denoise mặc định 0.6 (workflow Forge cũng 0.6).
- **Forge `flow_guard`** (mặc định 4px): căn flow chỉ kéo về chỗ AI trôi < guard, chỗ lệch > 2 x guard giữ nguyên hình AI (không kéo giãn). Log % ảnh bị chặn + cảnh báo khi p95 > 5px. 0 = như cũ.
- Không làm structure_lock cho Forge (kéo chi tiết về gốc = mất enhance, đúng vấn đề DB9U Upscale).
- `sampling.ksample` nhận `sigmas` truyền thẳng. Widget mới thêm ở CUỐI node (workflow đã lưu không lệch).
- Test: test_true_denoise, test_flow_guard.

## v0.12.0 — 2026-10-05
### ✅ Đã làm
- **DB9U Forge** (node mới, `db9u/forge.py`): node riêng gộp 2 bộ — tạo hình kiểu db9_flux_locked_upscale, ghép hình kiểu Ultimate.
  - Đọc lại workflow flux thật (`260531-Upscale_F2K9B-Ver_001`): ảnh -> UltraSharp tới 6000 -> TileSplit 2048 -> node chính `scale 2` (UltraSharp lần 2 trên từng ô, chia ô 2048 bên trong, denoise 0.3, 4 steps) -> TileMerge **thu 4096 về 2048**. Tức AI vẽ ở x2 kích thước ra rồi thu nhỏ = supersampling — lý do chính chi tiết bộ flux mịn/sắc hơn. Flux Match (v0.11.3) không có bước này.
  - Tạo hình: `supersample` 2.0/1.5/1.0, `ss_source` (chạy lại upscale model trên ô như flux / lanczos), khoá màu flux (chỉ mean/std + contrast của tần số thấp, HF giữ 100%, pad reflect), unsharp ở độ phân giải làm việc, `downscale` bicubic_sharp (như flux) / lanczos / area, `global_color` theo ảnh gốc, seed mỗi ô.
  - Ghép: plan_tiles (ô đều, bội 32) + merge_tiles (dốc tuyến tính đúng vùng giao), 1 tầng chia ô, căn trôi từng ô, QA ΔE + retry (mặc định 0), flow_align cả ảnh, chroma_lock / edge_guard tuỳ chọn, lô theo VRAM + tự hạ lô/ô khi OOM, resume cache.
  - Tối ưu: positive = negative mà cfg > 1 -> tự ép cfg 1 (workflow flux cũ chạy cfg 8 với pos = neg = tốn gấp đôi, kết quả y hệt).
- Test: `tests/test_forge_mock.py` (kích thước, supersample 1/1.5/2, lô, ép cfg, OOM, retry + ref tile, khoá màu giữ HF, ghép không vết nối, workflow khớp node).
- **Workflow `workflows/DB9U_Forge_Flux2Klein9B.json`**: UNET/CLIP/VAE Klein 9B → LoRA upscale_mir 0.8 → LoRA epi_noiseoffset2 0.8 → prompt cũ (cắm cả positive lẫn negative) + 4x-UltraSharp → DB9U Forge (6K, denoise 0.3, supersample 2) → Finish + Log + ghi chú chỉnh nhanh.
- **Sửa Region Fix**: vùng sát mép phải/dưới của ảnh có cạnh không chia hết 32 bị lùi góc crop lệch lưới 32 (ảnh 600px → y0 = 312). Giờ góc crop luôn bội 32, cỡ crop ở mép có thể lẻ (run_pass pad canvas lên bội 32, stitch tự đưa patch về đúng cỡ). Test `test_region_stitch_exact` còn so hiệu 2 tổng float32 (sai số ~0.004 dù không pixel nào đổi) → đổi sang so max ngoài crop.
- Test `test_region_source_file` (lộ ra sau khi sửa test trên): ngưỡng jpg95 0.03 đo cả nửa ảnh nhiễu ngẫu nhiên từng pixel (JPEG 4:2:0 lệch ~0.07 ở đó, nửa mịn chỉ 0.003) -> đo nửa mịn, ngưỡng 0.01. Code lưu/đọc ảnh không lỗi.
### ⏳ Chưa kiểm chứng
- Chưa chạy với model thật; chưa so ảnh Forge với BBB-UpscaleF2K9B cùng seed/LoRA.

## v0.11.3 — 2026-10-02
### ✅ Đã làm
- **DB9U Flux Match** (node mới, `db9u/fluxmatch.py`): 1 thanh `flux_like` 0–1 kéo DB9U về kiểu bộ flux cũ (db9_flux_locked_upscale / BBB-UpscaleF2K9B). Nối Advanced Settings -> Flux Match -> settings. Nội suy structure_lock, edge_guard về 0, lock_sigma -> 32; mốc 0.25: material_denoise = denoise + reference strip · 0.5: scheduler simple + retry 0 · 0.75: denoise đều (lá, trời) + ô 2048/giao 192 · 1.0: seed mỗi ô. `keep_color` BẬT giữ khoá màu/chroma_lock; TẮT: chroma_lock/lock_strength trôi về bộ cũ, ở 1.0 dùng lab_stats 0.55 + tắt flow_align. Output `note`: mức đang chọn, ưu / nhược, đã đặt gì, việc cần chỉnh ngoài node (denoise 0.5, LoRA 0.8, UltraSharp, SR Refine).
- Sửa: `texture_denoise` = 0 giờ đúng nghĩa "tắt denoise theo vùng" (lá = denoise chính); trước đó bị hiểu là denoise 0 cho vùng lá.
- Workflow Klein + Klein Detail: thêm Flux Match (Klein 0 = không đổi, Detail 0.5) + ô Ghi chú + bảng ưu / nhược. Qwen: chưa thêm (node dành cho flux).
- Test: test_flux_match.
### ⏳ Chưa kiểm chứng
- Chưa chạy trên máy chủ; chưa so ảnh Flux Match 1.0 với BBB-UpscaleF2K9B. Không giống 100% được: bộ cũ khoá màu theo từng ô (Reinhard RGB), ghép mép 64px, unsharp cuối.

## v0.11.2 — 2026-10-02
### ✅ Đã làm
- Đo img_00007 (preview O8/O9/O7 × d0.35/0.45/0.55, profile klein_detail + LoRA + 4x-UltraSharp): màu mảng lớn (σ16) khớp gốc, nhưng màu ở chi tiết dải 2–8px gấp 3.5x gốc, 0.7–2px gấp 2.8x (độ sáng chỉ 1.4x) -> cành tím, lá hồng/vàng, vân đá xanh đỏ. 3 mức denoise gần giống nhau (chênh 7–12/255, so với gốc 23) -> phần lỗi chung nằm trước/ngoài bước denoise (SR model hoặc AI với ref tile + LoRA) — cần test tách.
- **`chroma_lock`** (mặc định 1.0, widget cuối Settings): sau fusion, giữ độ sáng (chi tiết, nếp, vân) của kết quả, lấy màu a/b Lab từ ẢNH GỐC phóng lanczos (không từ SR). Thử trên ô O8: màu dải 2–8px 6.98 -> 2.14 (gốc 2.04). 0 = như cũ. Áp cho preview, chạy full, Region Fix.
- Workflow mẫu (Klein, Qwen, Detail) thêm chroma_lock 1.0.
- Test: test_chroma_lock.
### ⏳ Chưa rõ
- Độ sáng vẫn gắt/vẽ lại cành (HDR) — cần chủ test tách: (a) bypass upscale model, (b) reference_mode strip, (c) bypass LoRA.

## v0.11.1 — 2026-10-01
### ✅ Đã làm
- **`preview_pick`** (menu, cuối Advanced Settings): ô đã chọn · bảng ô · 1 / 3 / 5 ô khó nhất · tất cả ô — không phải gõ preview_tiles nữa. Đổi menu tự bật preview_tile. "ô đã chọn" = ô bấm trên bảng (🔲 Chọn ô trên bảng; Áp dụng tự đặt menu về mục này). Workflow cũ chưa có widget -> chạy như trước.
- Workflow mẫu (Klein, Qwen, Klein Detail) thêm preview_pick; Detail = 3 ô khó nhất + bảng chỉnh đẩy chi tiết (material_denoise, structure_lock, lock_sigma, LoRA, Region Fix).
- Test: test_preview_pick_menu.

## v0.11.0 — 2026-10-01
### ✅ Đã làm
- Phân tích workflow cũ BBB-UpscaleF2K9B (db9_flux_locked_upscale, Klein 9B + LoRA DB9_Upscale_Ver004 0.8, denoise 0.5): chi tiết x2 bicubic (ưu điểm) nhưng ΔE cục bộ p95 ~13, lệch hình tới ~6px @6K, đổi vật liệu (đá trần, caustic hồ, mặt bàn). Nguyên nhân: structure guide không nối (tắt), denoise 0.5, khoá màu chỉ mean/std cả ô, cfg 3.5 với pos=neg (= cfg 1, tốn x2), seed khác mỗi ô, unsharp chồng UltraSharp. Chi tiết: project doc claude/Klein9B_old_workflow_analysis.md.
- **Profile `flux2_klein_detail`** (chọn tay trong Advanced Settings): như Klein distilled (4 steps, cfg 1, flux2) + `reference_mode` tile — latent ô gốc gắn vào reference_latents để Klein sửa ô thay vì vẽ lại. reference tile -> 1 ô/lượt.
- **Workflow test `workflows/DB9U_Flux2Klein9B_Detail.json`**: LoRA DB9_Upscale_Ver004 0.8 (clip 0) + prompt trigger của workflow cũ, 4x-UltraSharp bật, 6K, denoise 0.45, profile flux2_klein_detail, preview auto3 × denoise 0.35/0.45/0.55.
### ⏳ Chưa kiểm chứng
- Chưa chạy trên máy chủ: VRAM/tốc độ khi có reference latent (chuỗi token gấp đôi) ở ô 1536; so ref tile vs strip cùng denoise.

## v0.10.1 — 2026-10-01
### ✅ Đã làm
- Lỗi: kéo ảnh 6K+ (PNG 16-bit) vào Load Image của Region Fix bị báo quá lớn (ComfyUI giới hạn upload, mặc định 100MB).
- **DB9U Region Source**: nạp ảnh lớn thẳng từ đĩa theo path (tuyệt đối hoặc trong output ComfyUI), đọc PNG/TIFF 16-bit bằng OpenCV — không upload. Tự tạo ảnh proxy nhỏ `input/db9u_mask_<tên>.jpg` (proxy_max 2048) để tô mask trong Load Image; Region Fix tự phóng mask về cỡ thật. Có input image để nối thẳng output Upscale. OUTPUT_NODE -> chạy riêng được để tạo proxy trước.
- Workflow mẫu: group Region Fix = Region Source (path) -> Region Fix; Load Image chỉ lấy MASK từ proxy.
- Test: test_region_source_file.

## v0.10.0 — 2026-10-01
### ✅ Đã làm
- **DB9U Region Fix** (tất cả trong 1): tô vùng chưa ưng bằng MaskEditor (hoặc gõ `x,y,w,h; ...`) -> chạy lại riêng vùng đó với denoise/prompt/seed/cfg khác -> ghép lại đúng từng pixel. Mỗi mảng tô rời = 1 vùng. Mask tô trên ảnh thu nhỏ vẫn dùng được (tự phóng). Dùng lại toàn bộ pipeline (chia ô nếu vùng lớn, denoise theo vùng, flow_align, enhance/structure_lock, QA ô) + Advanced Settings chung.
  - Ghép chuẩn: toạ độ crop số nguyên bội số 32 lưu trong REGION_INFO; `padding` = ngữ cảnh cho model, KHÔNG dán; mép hoà bằng mask feather, ép = 0 ở viền crop; `color_match` khoá màu/mảng lớn theo vùng cũ. `work_scale` 1-2: phóng vùng nhỏ lên cho model vẽ chi tiết rồi thu về đúng cỡ.
  - QA mỗi vùng: ΔE trong vùng + ΔE dải viền hoà (cảnh báo nếu > 4). Output before_after (trước | sau) + region_mask.
- **DB9U Region Crop / Region Stitch**: cặp tách rời để tự chạy KSampler/inpaint ngoài rồi ghép lại (nhiều mảng -> gộp 1 vùng).
- **GPU chạy đầy hơn**: `async_qa` (mặc định bật, widget cuối Settings) — căn trôi + ΔE + lưu cache ô chạy ở luồng phụ song song lúc GPU sample lô kế; ô lệch chạy lại sau cùng. Kết quả y hệt tuần tự.
- Log chẩn đoán: sau lô đầu báo model nạp bao nhiêu GB lên GPU (không đủ -> cảnh báo offload PCIe, nguyên nhân chính GPU không chạy hết công suất; đủ + VRAM trống > 4GB -> gợi ý batch_tiles 2); cuối lượt báo % thời gian GPU bận; thời gian bước khoá/fusion CPU sau lượt.
- Workflow mẫu: thêm group Region Fix (Load Image + MaskEditor -> Region Fix -> Save / Trước|Sau / Log) nối sẵn model, VAE, prompt, ControlNet, Settings; Settings thêm async_qa.
- Test: test_region_stitch_exact, test_region_fix_node, test_async_qa_same_result.

## v0.9.1 — 2026-10-01
### ✅ Đã làm
- Đo img_00012 (enhance 1) so với ảnh gốc phóng bicubic: hình học lệch rất ít (optical flow p50 0.27px, p90 0.78px), NHƯNG AI vẽ lại hình ở vùng lá/cỏ/đá: tương quan chi tiết mịn dương xỉ 0.44, cỏ 0.47, cột đá 0.56 (tường/trần/sàn 0.72-0.73); vân vừa dương xỉ/cỏ 0.68. Cột đá thô thành vân như đá marble, dương xỉ/cỏ đổi dáng lá.
- Nguyên nhân: enhance 1 nhận hết dải vân vừa + mịn của AI; vùng lá còn khoá màu ở mảng x4 (32px) -> AI giữ cả dáng lá tới 32px; denoise lá +0.2.
- **`structure_lock`** (mặc định 0.5): cổng tương quan theo 3 dải — mịn (<detail_sigma), vừa (detail_sigma..lock_sigma), thô vùng lá (lock_sigma..x texture_lock_mult). Dải nào AI không khớp cấu trúc gốc thì lấy từ SR (mịn/vừa) hoặc ảnh vào (thô). Thử offline trên crop img_00012: cỏ/dương xỉ về dáng gốc, tường/trần/sàn giữ gần hết chi tiết AI (cổng vừa 0.9-1.0).
- Log in % dải AI được nhận.
- Workflow mẫu: thêm structure_lock 0.5 + ghi chú.
- Test: test_enhance_structure_lock.

## v0.9.0 — 2026-10-01
### ✅ Đã làm
- So với repo db9_flux_locked_upscale (enhance tốt): repo cũ giữ TOÀN BỘ chi tiết AI, chỉ sửa màu/tương phản ở tần số thấp (avg_pool 15px) + unsharp cuối; workflow dùng LoRA upscale riêng 0.8 + denoise 0.3. DB9U v0.6+ (band fusion) bỏ vân vừa của AI, thay bằng vân SR, chi tiết mịn AI còn bị cổng tương quan x0.6 cắt -> 'upscale ổn nhưng không enhance'.
- **`enhance`** (Advanced Settings, mặc định 1.0): 1 = giữ vân + chi tiết AI, chỉ khoá màu/mảng > lock_sigma theo ảnh vào (vùng lá khoá ở mảng lớn hơn x texture_lock_mult); flow_align vẫn chạy trước để không bóng mờ; edge_guard giữ viền kiến trúc. 0 = band fusion cũ; giữa = trộn.
- DB9U Finish: ảnh TRƯỚC trên node thu cùng độ phân giải với ảnh SAU (trước đây trước 2048px, sau 720px -> sau trông mờ hơn); thumb 1280, bản nháp khi kéo 1400.
- Workflow mẫu: thêm `enhance` = 1 vào Settings + ghi chú.
- Test: test_with_upscale_model kiểm cả enhance 1 / 0 / 0.5.

## v0.8.3 — 2026-10-01
### ✅ Đã làm
- SR Refine chỉnh thông số không chạy lại quy trình: ComfyUI giữ cache node phía trước (seed DB9U Upscale để `fixed`); node giữ cache kết quả model (2 bản gần nhất, khoá theo ảnh + model + ô) -> đổi strength/mode/detail_sigma chỉ trộn lại. Đổi `tile`, model hoặc ảnh vào mới chạy model.
- Test: test_sr_refine_cache.
- Workflow mẫu (Qwen21, Flux2Klein9B): thêm DB9U SR Refine (Upscale → Refine → QAQC/Finish, dùng chung Load Upscale Model) + DB9U Advanced Settings nối sẵn (mặc định, preview_layout frame); ghi chú cập nhật.

## v0.8.2 — 2026-10-01
### ✅ Đã làm
- Node mới **DB9U SR Refine**: chạy ảnh qua upscale model theo ô, thu từng ô về cỡ cũ bằng area (supersampling) -> nét hơn, KHÔNG tăng kích thước, không giữ ảnh x4 trong RAM. `detail_only`: chỉ cộng chênh lệch chi tiết mịn (sigma `detail_sigma`) giữa ảnh model và ảnh vào, chặn ±0.12 chống quầng; `full`: trộn thẳng. Hết VRAM tự giảm ô.
- Test: test_sr_refine.

## v0.8.1 — 2026-10-01
### ✅ Đã làm
- Kiểm img_00007 (preview O3,O7,O9, 6K): màu khớp gốc (lệch trung bình RGB <0.01), form/viền không lệch; nhưng ảnh là bảng ghép 2888x2696 -> không so được với ảnh gốc.
- **preview_layout = frame** (mặc định): preview trả về cả khung ảnh ở kích thước cuối, ô preview đặt đúng vị trí (ghép mép mềm như chạy full), phần còn lại là ảnh phóng thường; original_resized = gốc cùng khung -> so sánh trực tiếp. `grid` giữ kiểu bảng cũ; so nhiều denoise luôn dùng grid.
- Widget mới thêm ở CUỐI Advanced Settings (không làm lệch giá trị node đã tạo).
- Test: test_preview_frame.

## v0.8.0 — 2026-09-30
### ✅ Đã làm
- **Bấm chọn ô trên bảng** (`web/js/db9u_tiles.js`): nút `🔲 Chọn ô trên bảng` trên DB9U Upscale + Advanced Settings; bảng ô gửi toạ độ ô lên giao diện (`db9u_board`). Bấm chọn/bỏ (vùng giao lấy ô tâm gần nhất), "3 ô khó nhất", nhập so denoise, Áp dụng / Áp dụng & chạy (tự bật preview_tile).
- **So nhiều denoise trên 1 ô**: `preview_denoise` "0.3,0.4,0.5" → hàng = ô, cột = denoise.
- **Dùng lại ô preview khi chạy full** (`reuse_preview`, cần resume, chỉ ảnh 1 lượt): ô preview ở denoise chính qua đúng QA như chạy full (căn trôi + ΔE + retry) rồi lưu cache; run_pass nạp lại. Preview và full dùng chung cách tạo ảnh nền (SR half → resize) để ô khớp.
- **despeckle** (0 = tắt): khử đốm đơn lẻ (median 3x3 theo độ sáng, giữ màu) + hạt mịn (guided filter) ở mảng phẳng của ảnh gốc trước khi phóng; bỏ qua lá/texture dày và cạnh kiến trúc.
- Refactor: `_qa_tile` dùng chung preview/full; `mosaic(cols=)`.
- DB9U Upscale trả `{"ui","result"}`.
- Test: test_board_ui, test_preview_denoise_grid, test_reuse_preview_in_full_run, test_despeckle.
### ⏳ Chưa kiểm chứng
- Chưa chạy test/ComfyUI thật (máy phát triển không có torch). Cần: `tests\test_engine_mock.py` ALL PASS + thử bảng ô bấm chọn trên giao diện + despeckle 0.3–0.5 trên TU2119_Cam_002.
- Node cũ trong workflow: xoá DB9U Upscale / Advanced Settings rồi thêm lại để có nút + widget mới.

## v0.7.0 — 2026-09-30
### ✅ Đã làm
- **Bảng ô**: `preview_tile` bật + `preview_tiles` trống -> chỉ vẽ bảng ô O1..On (khung + tên + % chi tiết, 3 ô khó nhất khung cam), không chạy model.
- **Chạy thử ô chọn**: `preview_tiles` = `O1,O5,O6` / `O3-O6` / `auto` / `auto3` -> chỉ chạy các ô đó ở kích thước cuối, ghép thành 1 bảng có nhãn (ảnh sau / ảnh trước / zone cùng bố cục).
- **Editor không chạy lại workflow**: chỉnh màu port 1:1 công thức Python sang JS, chạy trong Web Worker (bản nháp ≤900px khi kéo, bản đủ khi thả). Chỉ 'Lưu full-res' mới chạy lệnh (Python, chính xác).
- **So sánh bật/tắt**: nút trên node + trong Editor (phím Y), giữ `\` xem ảnh trước, kéo vạch chia, lăn zoom, kéo để pan, double-click / Z: vừa khung ↔ 1:1.
- **Giao diện kiểu Lightroom**: histogram RGB, nhóm Basic (WB/Tông/Presence) với thanh màu, Tone Curve kéo điểm trực tiếp (RGB/R/G/B, preset), HSL tab Hue/Sat/Lum, Color Grading tab Shadows/Mid/High, Selective, Hiệu ứng, LUT (đọc .cube qua `/db9u/lut`), Xuất file; mỗi nhóm có 'Đặt lại'.
- Finish gửi thêm ảnh preview chưa chỉnh (`db9u_raw`) cho Editor.
### ⏳ Chưa làm
- Grain / Texture trên Editor là xấp xỉ (nhiễu ngẫu nhiên khác seed; texture quy đổi theo tỉ lệ preview) — bản lưu dùng công thức Python.

## v0.6.1 — 2026-09-30
### ✅ Đã làm
- Sửa preview_tile ra ảnh méo vuông + nổi vân: QAQC so ô preview với CẢ ảnh gốc rồi auto_fix đè màu ảnh gốc bị bóp vào ô. QAQC giờ tự bỏ qua khi khác tỉ lệ.
- Workflow mẫu: QAQC lấy `original_resized` của DB9U Upscale (đúng cả chế độ preview lẫn full).
- `auto_fix` mặc định TẮT: khoá màu+viền sigma ~2-3px theo ảnh gốc bicubic làm mềm chi tiết vừa enhance.
- Test: test_qaqc_aspect_guard.

## v0.6.0 — 2026-09-30
### ✅ Đã làm
- Đo trên ảnh thật (TU2119_Cam_002 -> img_00005, 6K): vùng lá/đá cho AI trọn quyền -> quầng sáng mờ trên lá, vân đá bị quệt, kém cả bicubic ảnh gốc.
- **Band fusion cho MỌI vùng** (kể cả lá/đá): mảng lớn gốc + vân vừa SR/gốc + chi tiết mịn AI qua cổng tương quan. `texture_ai` (mặc định 0) nếu muốn lá vẽ lại nhiều hơn.
- **flow_align**: căn ảnh AI về vị trí gốc từng pixel (OpenCV DIS optical flow, giới hạn ±16px) trước khi trộn -> hết bóng mờ do lệch lớp. Thiếu OpenCV thì tự bỏ qua, log báo.
- Không nối upscale model vẫn chạy band fusion (dùng ảnh phóng lanczos làm nguồn vân).
- Test: test_flow_align.
### ⏳ Chưa làm
- Đốm lấm tấm trên tường vữa có sẵn trong ảnh gốc (render noise) — node đang giữ đúng gốc; muốn xoá cần bước khử nhiễu riêng.

## v0.5.0 — 2026-09-30
### ✅ Đã làm
- **Band fusion** (cần nối upscale_model): tách 3 dải tần sau mỗi lượt — mảng lớn = ảnh gốc, vân vừa = SR model của ẢNH GỐC (vân vật liệu thật), chi tiết mịn = SR + AI theo `ai_detail`. Tường/đá/gỗ không còn bị thay bằng vân AI.
- **fidelity_gate**: đo tương quan cục bộ giữa chi tiết AI và chi tiết SR; chỗ AI bịa (đốm/hạt trên tường phẳng) tự bỏ phần AI.
- Upscale model chạy 1 lần trên ảnh gốc và làm nguồn chi tiết cho MỌI lượt (trước chỉ lượt 1, lượt 2 dùng lanczos -> mềm). Lượt ≥2: giữ khối của lượt trước + chi tiết mịn từ SR gốc.
- Vùng **vật liệu** có denoise riêng `material_denoise` (auto = min(denoise, 0.4)); lá/cỏ vẫn denoise+0.2.
- Settings mới (cuối node): material_denoise, ai_detail, detail_sigma, fidelity_gate.
- Workflow mẫu: bật sẵn node Load Upscale Model.
- Test: test_band_fusion, test_with_upscale_model.
### ⏳ Chưa làm
- Chưa kiểm trên GPU thật; cần chạy test_engine_mock.py và 1 ảnh.

## v0.4.3 — 2026-09-29
### ✅ Đã làm
- Sửa lá/cỏ bị nhoè, có quầng sáng "bóng ma": khoá màu detail_transfer ở vùng texture dày giờ chỉ khoá mảng lớn (lock_sigma × `texture_lock_mult`, mặc định 4). Trước đây đốm sáng lá của ảnh gốc bị đè lên lá mới model vẽ lại.
- Thêm `texture_lock_mult` vào DB9U Advanced Settings (cuối danh sách, không lệch workflow cũ). 1 = như bản cũ.
- Test: `test_color_lock_zoned`.

## v0.4.2 — 2026-09-29
### ✅ Đã làm
- Sửa lỗi JS không nạp (sai đường import `../../` → `../../../scripts/app.js` vì file nằm trong web/js) → node Finish giờ chỉ hiện khung so sánh + nút Editor/Lưu.
- Ẩn widget chắc hơn (thêm `hidden`) cho frontend mới.

## v0.4.1 — 2026-09-29
### ✅ Đã làm
- DB9U Finish: node chỉ còn khung so sánh trước/sau (rê chuột) + 2 nút "Mở Editor" / "Lưu full-res"; toàn bộ thông số ẩn khỏi node.
- Editor toàn màn hình kiểu Lightroom/Camera Raw: ảnh lớn so sánh A/B (rê chuột, lăn zoom, chuột phải/giữa kéo, giữ \ xem ảnh trước, 100%/vừa khung) + panel: Basic, Cân bằng trắng, Presence, Tone Curve (preset S-curve/Matte), HSL (tab Hue/Sat/Lum), Color Balance (tab Shadows/Mid/High), Selective Color, Effects, LUT, Xuất file; double-click thanh = về mặc định; Reset tất cả.
- Mỗi lần chỉnh tự chạy lại riêng node trên preview; Lưu full-res xong tự quay về chế độ preview.
### ⏳ Chưa kiểm chứng
- Chưa thử trên giao diện thật (JS chỉ kiểm cú pháp). Chế độ Nodes 2.0 có thể không vẽ khung so sánh trên node (Editor vẫn dùng được).

## v0.4.0 — 2026-09-29
### ✅ Đã làm
- Node **DB9U Finish** = chỉnh màu (Basic, Presence, Curves, HSL, Color Balance, Selective Color, LUT) + so sánh trước/sau (thanh trượt vẽ ngay trên node, rê chuột) + lưu file, trong 1 node.
- save_file tắt: chỉ chỉnh trên preview (nhanh); live_edit: đổi thông số là tự chạy lại (debounce 0.45s). save_file bật: áp full-res + lưu.
- JS frontend web/js/db9u_finish.js (WEB_DIRECTORY). Workflow thay Save + Compare Images + Basic/Presence bằng DB9U Finish.
### ⏳ Chưa làm / chưa kiểm chứng
- JS chưa thử trên giao diện thật (vẽ trên canvas kiểu node cũ; chế độ Nodes 2.0 có thể không hiện thanh trượt).
- test_finish_node chờ chạy trên máy.

## v0.3.0 — 2026-09-29
### ✅ Đã làm
- P3 Color Grade (7 node, menu DB9 Ultimate/Color): Basic (exposure, contrast, highlights, shadows, whites, blacks, temperature, tint, vibrance, saturation) · Presence (clarity, texture, dehaze dark-channel, vignette, grain) · Curves (monotone cubic, master/R/G/B) · HSL 8 dải màu · Color Balance · Selective Color (xấp xỉ Photoshop) · LUT .cube (thư mục luts/ hoặc đường dẫn).
- P4: zone map (lá/texture denoise cao, trời tự nhận ~0.12, kiến trúc = denoise) · resume (lưu ô đã xong vào output/db9u_cache, khoá theo ảnh + prompt + model/LoRA + thông số) · preview_tile (chạy 1 ô nhiều chi tiết nhất) · notebook Colab.
- Review agent: sửa ô đồng nhất chạy sai denoise, cache lệch plan, khoá cache thiếu, grain chỉ sáng, curve 1 điểm, selective color relative, nâng sáng pixel đen.
- Workflow: output zone_map, chuỗi Basic → Presence (bypass).
### ⏳ Chưa làm / chưa kiểm chứng
- Chưa chạy test mới trên máy (test_engine_mock: preview/resume/grade).
- Không làm: tự viết caption (chưa kiểm chứng node TextGenerate nhận ảnh), đo vết nối trong QAQC, preset JSON cho Color Grade (workflow ComfyUI đã lưu giá trị).
- Nhận trời là heuristic (vùng phẳng sáng nối mép trên); không nhận kính/nước.

## v0.2.0 — 2026-09-29
### ✅ Đã làm
- Node DB9U Save: lưu vào local_folder hoặc output ComfyUI (tự nhận cloud Colab/Kaggle/RunPod), png16/tiff16 (opencv) · png8 · jpg95, đánh số file tự động.
- Xuất cặp preview (≤3072px) cho node gốc "Compare Images" của ComfyUI → so sánh trước/sau bằng thanh trượt.
- Workflow cập nhật: DB9U Save + Compare Images.
### ⏳ Chưa làm / chưa kiểm chứng
- test_engine_mock.py thêm test_save_and_preview — chờ chạy trên máy.

## v0.1.0 — 2026-09-29
### ✅ Đã làm
- P0+P1: core (port từ DB9_UpscaleEnchanceIMAGE v0.5.0) + control maps (tile/canny/lineart/grayscale).
- Model Profile JSON: qwen_image_21, flux2_klein_distilled, flux2_klein_base, generic; tự nhận model theo class.
- Hardware Profile: cỡ ô theo VRAM, lô ô trên GPU ≥40GB, fallback khi hết VRAM.
- Engine: nhiều lượt ≤2x, chia lô, căn ô trôi, QA + chạy lại ô lệch, denoise theo vùng, khoá màu Lab, khoá cạnh kiến trúc.
- ControlNet chuẩn ComfyUI (tự tắt khi không tương thích) + Qwen 2.1 Fun; Flux2Scheduler sigmas.
- Nodes: DB9U Upscale, DB9U Advanced Settings, DB9U QAQC. Workflow Qwen21 + Flux2 Klein 9B.
- tests/test_engine_mock.py ALL PASS trên máy chủ (ComfyUI 0.37.0, RTX 5070 Ti).
### ⏳ Chưa làm / chưa kiểm chứng
- tests/test_core.py: đã sửa test edge_guard, chờ chạy lại.
- Chưa chạy với model thật (Qwen 2.1 / Flux 2 Klein); lô nhiều ô trên A100/H100 chưa thử.
- P2 Save & Compare, P3 Color Grade, P4 preview/resume/zone map/caption.
