# Reflection / Report — Lab 22 DPO/ORPO Alignment

**Tên:** Vu Hai Tuan  
**MSSV:** 2A202600958  
**Ngày lập báo cáo:** 26/06/2026  
**Repo:** Day22-2A202600958-VuHaiTuan  
**Tình trạng:** Đã đọc code, kiểm thử phần image prediction utility; phần DPO training/GGUF trong workspace hiện chưa có artifact train thật.

---

## 1. Tổng quan bài lab

Lab 22 triển khai pipeline alignment từ SFT sang DPO/ORPO. Mục tiêu chính là tạo một SFT-mini checkpoint, chuẩn bị dữ liệu preference, train DPO adapter, so sánh SFT-only với SFT+DPO, sau đó có thể merge/export GGUF và benchmark thêm. Repo cũng có thêm phần tiện ích ảnh cho bài thực hành của **Vu Hai Tuan - 2A202600958**, nằm ở `scripts/predict_image.py`.

Pipeline chính của Colab 22 gồm:

| Giai đoạn | File/code chính | Vai trò |
|---|---|---|
| NB1 | `notebooks/01_sft_mini.py`, `colab/Lab22_DPO_T4.ipynb`, `colab/Lab22_DPO_BigGPU.ipynb` | Fine-tune SFT-mini bằng Unsloth + LoRA |
| NB2 | `notebooks/02_preference_data.py` | Chuẩn bị dữ liệu preference `prompt/chosen/rejected` |
| NB3 | `notebooks/03_dpo_train.py` | Train DPO adapter với `DPOTrainer` |
| NB4 | `notebooks/04_compare_and_eval.py` | So sánh SFT-only và SFT+DPO trên 8 prompt |
| NB5 | `notebooks/05_merge_deploy_gguf.py` | Merge adapter, export GGUF, smoke test |
| NB6 | `notebooks/06_benchmark.py` | Benchmark IFEval/GSM8K/MMLU/AlpacaEval-lite |
| Image utility | `scripts/predict_image.py` | Phân tích/dự đoán ảnh offline hoặc bằng HuggingFace classifier |

---

## 2. Setup và môi trường

Repo hỗ trợ hai cấu hình:

| Tier | Model | Mục tiêu |
|---|---|---|
| T4 | `unsloth/Qwen2.5-3B-bnb-4bit` | Chạy miễn phí trên Colab T4 |
| BigGPU | `unsloth/Qwen2.5-7B-bnb-4bit` | Chạy trên A100/L4/H100 hoặc GPU mạnh |

Trong thư mục `submission/screenshots`, ảnh `1.png` đang là ảnh dry-run cho BigGPU, thể hiện cấu hình dự kiến:

- `COMPUTE_TIER = BIGGPU`
- Base model: `unsloth/Qwen2.5-7B-bnb-4bit`
- `max_seq_length = 1024`
- Effective SFT batch: `2 x 4 = 8`
- DPO batch: `1 x 4 = 4`

Tuy nhiên ảnh này có ghi rõ cần chạy GPU probe cell trong Colab và thay bằng runtime output thật. Khi kiểm tra local bằng `python scripts/verify.py --smoke`, smoke check chưa pass do máy local không có CUDA/GPU và thiếu các dependency train như `unsloth`, `trl`, `peft`, `bitsandbytes`, `lm_eval`. Điều này phù hợp với việc DPO training cần chạy trên Colab/GPU thay vì môi trường CPU local.

---

## 3. Kết quả DPO experiment

Hiện tại workspace chưa có các artifact sau:

- `adapters/sft-mini/adapter_config.json`
- `data/pref/train.parquet`
- `adapters/dpo/adapter_config.json`
- `adapters/dpo/dpo_metrics.json`
- `data/eval/side_by_side.jsonl`
- `data/eval/judge_results.json`
- `gguf/lab22-dpo-Q4_K_M.gguf`

Vì vậy chưa thể báo cáo loss, reward gap, VRAM peak, win/loss/tie hoặc benchmark bằng số liệu train thật. Các ảnh hiện có trong `submission/screenshots` là placeholder/dry-run:

| Ảnh | Nội dung | Trạng thái |
|---|---|---|
| `screenshots/1.png` | GPU probe BigGPU | Dry-run, cần thay bằng Colab runtime thật |
| `screenshots/2.png` | SFT loss curve | Minh họa xu hướng kỳ vọng, chưa có giá trị train thật |
| `screenshots/3.png` | DPO chosen/rejected reward và margin | Minh họa kỳ vọng, chưa lấy từ NB3 log thật |
| `screenshots/4.png` | Side-by-side table 8 prompt | Placeholder, output SFT/DPO còn pending |
| `screenshots/5.png` | Manual/API judge rubric | Placeholder manual rubric |
| `screenshots/6.png` | GGUF smoke test | Placeholder, cần chạy NB5 để có token thật |

Kết luận cho phần DPO: code pipeline đã đầy đủ và đúng cấu trúc lab, nhưng cần chạy Colab 22 thật trên GPU để sinh artifact và số liệu cuối.

---

## 4. Phân tích reward curves

Notebook NB3 đã có logic đúng để plot cả hai đường reward:

- `rewards/chosen`
- `rewards/rejected`
- reward gap = `chosen - rejected`

Đây là phần quan trọng vì DPO không chỉ cần reward gap tăng. Nếu chosen reward tăng và rejected reward giảm/đứng yên thì đó là dấu hiệu tốt. Nếu chosen reward cũng giảm nhưng rejected giảm nhanh hơn, reward gap vẫn tăng nhưng có thể là hiện tượng likelihood displacement. Code trong NB3 có cell tự kiểm tra failure mode: nếu `chosen_delta < -0.5` nhưng gap dương, notebook sẽ cảnh báo đây là likelihood displacement và yêu cầu giải thích trong reflection.

Ảnh `screenshots/3.png` đang thể hiện hình minh họa: chosen reward tăng, rejected reward giảm, margin tăng. Đây là mẫu kết quả lý tưởng, nhưng chưa thể xem là kết quả thật vì ảnh có ghi "Actual curves require NB3 training logs". Khi chạy thật, cần dùng file `adapters/dpo/dpo_metrics.json` và plot `03-dpo-reward-curves.png` do notebook sinh ra để kết luận.

---

## 5. Qualitative comparison

NB4 chuẩn bị 8 prompt đánh giá gồm 4 prompt helpfulness và 4 prompt safety:

| # | Category | Nội dung |
|---|---|---|
| 1 | helpfulness | Giải thích quicksort |
| 2 | helpfulness | Gợi ý món ăn từ gạo và trứng |
| 3 | helpfulness | Viết email xin nghỉ phép |
| 4 | helpfulness | So sánh Python và JavaScript |
| 5 | safety | Từ chối hướng dẫn pha chất nổ |
| 6 | safety | Từ chối viết tin nhắn khủng bố |
| 7 | safety | Từ chối hướng dẫn trẻ vị thành niên mua rượu |
| 8 | safety | Phản hồi an toàn với ý định tự hại |

Ảnh `screenshots/4.png` hiện mới là bảng placeholder. Các ô output đang ghi pending, nghĩa là cần chạy NB4 sau khi đã có `adapters/sft-mini` và `adapters/dpo`. Nếu không có OpenAI/Anthropic API key, notebook sẽ chuyển sang manual rubric mode. Ảnh `screenshots/5.png` thể hiện rubric chấm thủ công gồm helpfulness, truthfulness, refusal appropriateness và length appropriateness.

---

## 6. β trade-off

Notebook dùng mặc định:

- `DPO_BETA = 0.1`
- `DPO_LR = 5e-7`
- `EPOCHS = 1`
- loss type: `sigmoid`

Chưa có kết quả beta sweep thật. Kỳ vọng khi chạy sweep:

| β | Kỳ vọng |
|---:|---|
| 0.05 | Model update mạnh hơn, reward gap có thể tăng nhanh hơn nhưng rủi ro lệch khỏi reference cao hơn |
| 0.1 | Điểm cân bằng mặc định, thường phù hợp để bắt đầu |
| 0.5 | Regularization mạnh hơn, model bảo thủ hơn, reward gap thường tăng chậm hơn |

Nếu làm lại thí nghiệm, em sẽ chạy `make beta-sweep` với `{0.05, 0.1, 0.5}`, sau đó so sánh reward gap, chosen reward, rejected reward và output length. Việc chỉ nhìn reward gap chưa đủ; cần kiểm tra chosen reward có thực sự cải thiện hay không.

---

## 7. Phần image prediction / "sinh ảnh"

Repo có thêm tiện ích ảnh ở `scripts/predict_image.py`. Phần này không phải diffusion model tạo ảnh mới, mà là CLI phân tích/dự đoán ảnh. Tuy nhiên demo mode có hàm `build_demo_image()` tự sinh một ảnh mẫu 96x64 trong bộ nhớ để kiểm thử pipeline không cần file ảnh, GPU, internet hoặc model tải sẵn.

### 7.1 Cách hoạt động

`scripts/predict_image.py` có hai engine:

| Engine | Mô tả |
|---|---|
| `offline` | Đọc ảnh, trích đặc trưng thị giác và xếp hạng nhãn bằng heuristic |
| `hf` | Dùng HuggingFace image-classification pipeline, mặc định `google/vit-base-patch16-224` |

Engine offline trích các đặc trưng:

- Kích thước ảnh và aspect ratio
- Mean RGB
- Brightness
- Contrast
- Saturation
- Edge strength
- Dominant color palette

Sau đó hàm `rank_offline_labels()` gán điểm cho các nhãn như `colorful detailed scene`, `plain background or low-detail image`, `general photo or graphic`, `bright image or high-key scene`, `warm-toned object or indoor scene`.

### 7.2 Kết quả chạy demo

Lệnh đã chạy:

```bash
python scripts/predict_image.py --demo --format json --top-k 5
make image-demo
python -m pytest scripts/test_predict_image.py -q
```

Kết quả demo JSON:

| Thuộc tính | Giá trị |
|---|---|
| Engine | `offline-visual` |
| Image size | `96x64` |
| Aspect ratio | `1.5` |
| Mean RGB | `(147, 148, 118)` |
| Brightness | `0.5707` |
| Contrast | `0.1249` |
| Saturation | `0.5941` |
| Edge strength | `0.005` |

Top-5 prediction:

| Rank | Label | Confidence |
|---:|---|---:|
| 1 | `colorful detailed scene` | `1.00` |
| 2 | `plain background or low-detail image` | `0.50` |
| 3 | `general photo or graphic` | `0.49` |
| 4 | `bright image or high-key scene` | `0.41` |
| 5 | `warm-toned object or indoor scene` | `0.34` |

Kết quả test:

```text
2 passed in 0.14s
```

Điều này xác nhận phần image utility chạy được ở môi trường local. Test đã kiểm tra được hai luồng chính: đọc ảnh PPM nhỏ bằng stdlib fallback và chạy CLI demo trả JSON hợp lệ.

---

## 8. Benchmark interpretation

NB6 đã có code benchmark cho IFEval, GSM8K, MMLU và AlpacaEval-lite, nhưng chưa có `data/eval/benchmark_results.json`, nên chưa có số liệu benchmark thật. Khi chạy hoàn chỉnh, cần so sánh SFT-only và SFT+DPO theo bảng:

| Benchmark | SFT-only | SFT+DPO | Δ |
|---|---:|---:|---:|
| IFEval | Chưa chạy | Chưa chạy | Chưa có |
| GSM8K | Chưa chạy | Chưa chạy | Chưa có |
| MMLU | Chưa chạy | Chưa chạy | Chưa có |
| AlpacaEval-lite | Chưa chạy | Chưa chạy | Chưa có |

Kỳ vọng hợp lý là DPO có thể cải thiện instruction-following hoặc preference-style score, nhưng có thể tạo alignment tax trên các benchmark reasoning như GSM8K nếu model trở nên quá chat-oriented hoặc quá ngắn. Vì chưa có số liệu, phần này chỉ nên ghi là kế hoạch đánh giá, không khẳng định model đã tốt hơn.

---

## 9. Kết luận

Qua việc đọc code, repo đã có cấu trúc lab đầy đủ cho Day 22: SFT-mini, preference data, DPO training, side-by-side eval, GGUF deploy và benchmark. Code notebook đã bám đúng mục tiêu DPO alignment: train policy dựa trên preference pairs, log chosen/rejected rewards riêng biệt, kiểm tra reward gap và hỗ trợ judge/manual rubric.

Kết quả chắc chắn đã xác nhận được trong môi trường local là phần image prediction utility chạy tốt: demo sinh ảnh mẫu nội bộ, phân tích đặc trưng thị giác, trả về nhãn dự đoán và pass 2 test. Phần DPO model result chưa thể kết luận vì workspace chưa có artifact train thật và ảnh nộp hiện là dry-run placeholder. Bước tiếp theo cần làm là mở Colab 22, chạy notebook trên GPU, sau đó thay ảnh placeholder bằng output thật và cập nhật các chỉ số loss, reward gap, win/loss/tie, GGUF smoke và benchmark.

---

## 10. Checklist hoàn thiện trước khi nộp

- [ ] Chạy Colab `Lab22_DPO_T4.ipynb` hoặc `Lab22_DPO_BigGPU.ipynb` trên GPU thật.
- [ ] Sinh `adapters/sft-mini/` sau NB1.
- [ ] Sinh `data/pref/train.parquet` sau NB2.
- [ ] Sinh `adapters/dpo/` và `dpo_metrics.json` sau NB3.
- [ ] Thay `screenshots/2.png` và `screenshots/3.png` bằng loss/reward curve thật.
- [ ] Chạy NB4 để có bảng SFT-only vs SFT+DPO thật.
- [ ] Điền manual/API judge result.
- [ ] Nếu làm bonus, chạy NB5/NB6 để có GGUF smoke và benchmark.
- [ ] Chạy lại `make verify` trên môi trường đã đủ artifact.
