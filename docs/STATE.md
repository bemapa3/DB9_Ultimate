# STATE — DB9_Ultimate
- Version: v0.12.4 (2026-10-06) — đã commit, tag, push.
- Xong: DB9U Forge (tạo hình kiểu flux + ghép kiểu Ultimate). Chốt mặc định Klein 9B: denoise 0.45, ref strip, sharpen 0.15, ss_source lanczos, edge_guard 0.5.
- Kiểm chứng thật: full 4000x2250 -> 6144x3456 (img_00027), 60 ô 11.2 phút, ΔE TB 1.88, flow p95 0.57px. Cây/đá thêm chi tiết, không lưới.
- 2026-10-06: thêm `docs/THONG_SO.md` (sổ tay thông số + công thức đẩy chi tiết lá/vải). Mục ⚠️ chưa test số liệu.
- Bước tiếp: chủ test công thức "đẩy chi tiết lá/vải" (Forge denoise 0.5–0.55 + chroma_lock 0.4) bằng preview auto3, báo kết quả để chốt vào sổ tay.
- Ý tưởng sau: sàn gỗ/vùng phẳng ΔE ~0.9 (ít enhance) — cân nhắc denoise theo độ chi tiết ô (Forge chưa có denoise theo vùng như DB9U Upscale).
