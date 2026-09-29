# Changelog

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
