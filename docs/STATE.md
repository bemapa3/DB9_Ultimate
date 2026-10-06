# STATE — DB9_Ultimate
- Version: v0.12.5 (2026-10-06) — commit, tag, push.
- Xong: DB9U Forge (tạo hình kiểu flux + ghép kiểu Ultimate). 2 bộ chốt Klein 9B trong `docs/THONG_SO.md` mục 0: 🅤 UPSCALE (ref tile, denoise 0.55) · 🅔 ENHANCE (ref strip, denoise 0.55, chủ duyệt).
- v0.12.5: nút 🗑 Reset preview & chạy lại (Forge/Upscale/Settings) — xoá `output/db9u_cache` + IS_CHANGED token. Test mock pass; ⏳ chưa bấm thử trên ComfyUI thật.
- Kiểm chứng thật trước: full 4000x2250 -> 6144x3456 (img_00027), 60 ô 11.2 phút, ΔE TB 1.88 (denoise 0.45); preview 9 ô denoise 0.55 ΔE 2.98.

## Bản đồ module
| Module | Trạng thái |
|---|---|
| Forge (tạo hình + ghép) | 🔒 |
| Preview + chọn ô | 🔒 |
| Reset preview (route + nút JS) | 🔧 chờ test tay |

- Bước tiếp: chủ pull v0.12.5 vào `custom_nodes/DB9_Ultimate`, khởi động lại ComfyUI, test nút Reset (checklist trong báo cáo).
- Ý tưởng sau: sàn gỗ/vùng phẳng ΔE ~0.9 (ít enhance) — cân nhắc denoise theo độ chi tiết ô (Forge chưa có denoise theo vùng như DB9U Upscale).
- File vừa sửa: db9u/engine.py, db9u/nodes.py, db9u/forge.py, web/js/db9u_tiles.js, tests/test_engine_mock.py, docs/THONG_SO.md, README.md, CHANGELOG.md, versions.json, __init__.py.
