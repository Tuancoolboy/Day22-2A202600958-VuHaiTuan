# Báo cáo kết quả Lab 22 — Vu Hai Tuan — 2A202600958

**Sinh viên:** Vu Hai Tuan  
**MSSV:** 2A202600958  
**Ngày:** 26/06/2026  
**Chủ đề:** DPO/ORPO Alignment và tiện ích image prediction trong Colab 22

## 1. Mục tiêu

Bài Lab 22 xây dựng pipeline alignment cho mô hình ngôn ngữ: tạo SFT-mini checkpoint, chuẩn bị dữ liệu preference, train DPO adapter, so sánh SFT-only với SFT+DPO, sau đó có thể merge sang GGUF và benchmark. Repo cũng bổ sung phần ảnh ở `scripts/predict_image.py` cho bài thực hành của Vu Hai Tuan.

## 2. Kết quả đọc code Colab 22

Colab 22 gồm 6 giai đoạn:

| Giai đoạn | Nội dung |
|---|---|
| NB1 | SFT-mini bằng Unsloth + LoRA, dataset `bkai-foundation-models/vi-alpaca` |
| NB2 | Format UltraFeedback thành `prompt/chosen/rejected` |
| NB3 | Train DPO bằng `DPOTrainer`, `beta=0.1`, `lr=5e-7` |
| NB4 | So sánh 8 prompt giữa SFT-only và SFT+DPO |
| NB5 | Merge/export GGUF Q4_K_M |
| NB6 | Benchmark IFEval/GSM8K/MMLU/AlpacaEval-lite |

Code có logic quan trọng để plot riêng `rewards/chosen`, `rewards/rejected` và reward gap. Đây là điểm đúng vì chỉ nhìn reward gap có thể bỏ sót hiện tượng likelihood displacement.

## 3. Tình trạng artifact

Workspace hiện chưa có artifact train thật như `adapters/sft-mini`, `adapters/dpo`, `data/pref/train.parquet`, `data/eval/side_by_side.jsonl` hoặc GGUF. Các ảnh trong `submission/screenshots` đang là dry-run/placeholder, có ghi rõ cần chạy notebook thật để lấy số liệu. Vì vậy chưa thể kết luận loss, reward gap, win-rate hoặc benchmark của model.

## 4. Kết quả phần ảnh

Phần ảnh trong repo là image prediction utility, không phải diffusion sinh ảnh. Demo mode tự tạo một ảnh mẫu 96x64 để test pipeline.

Lệnh đã chạy:

```bash
python scripts/predict_image.py --demo --format json --top-k 5
make image-demo
python -m pytest scripts/test_predict_image.py -q
```

Kết quả chính:

| Metric | Giá trị |
|---|---|
| Engine | `offline-visual` |
| Image size | `96x64` |
| Aspect ratio | `1.5` |
| Mean RGB | `(147, 148, 118)` |
| Brightness | `0.5707` |
| Contrast | `0.1249` |
| Saturation | `0.5941` |
| Edge strength | `0.005` |

Top prediction:

| Rank | Label | Confidence |
|---:|---|---:|
| 1 | `colorful detailed scene` | `1.00` |
| 2 | `plain background or low-detail image` | `0.50` |
| 3 | `general photo or graphic` | `0.49` |
| 4 | `bright image or high-key scene` | `0.41` |
| 5 | `warm-toned object or indoor scene` | `0.34` |

Pytest result:

```text
2 passed in 0.14s
```

## 5. Kết luận

Code Lab 22 đã có cấu trúc đầy đủ cho pipeline DPO alignment và có phần image prediction chạy được local. Phần kết quả chắc chắn hiện tại là image utility đã pass test và trả prediction hợp lệ. Phần DPO/GGUF/benchmark cần chạy Colab 22 trên GPU thật để sinh artifact và thay ảnh placeholder bằng kết quả thật trước khi nộp.
