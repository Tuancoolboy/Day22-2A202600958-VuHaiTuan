# Bao cao Lab 22 DPO Alignment — Vu Hai Tuan — 2A202600958

**Sinh vien:** Vu Hai Tuan  
**MSSV:** 2A202600958  
**Ngay chay:** 27/06/2026  
**Notebook:** `Lab22_DPO_Light.ipynb`  
**Che do:** LIGHT / LOW_MEM

## 1. Tom tat

Em da chay thanh cong pipeline Lab 22 theo ban Light de giam RAM, tranh loi kernel va dam bao notebook chay het. Ban Light khong dung model 7B/3B ma dung `Qwen/Qwen2.5-0.5B-Instruct`, LoRA rank 8, SFT slice 300 mau va preference slice 500 cap. Pipeline da tao SFT adapter, preference data, DPO adapter, bang so sanh 8 prompt, merged-light model va cac screenshot trong `submission/screenshots/`.

## 2. Moi truong va cau hinh

| Hang muc | Gia tri |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB |
| CUDA | 13.0 |
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| dtype | `torch.bfloat16` |
| Max length | 384 |
| Max prompt length | 192 |
| SFT slice | 300 |
| Preference slice | 500 |
| Batch | 1 |
| Gradient accumulation | 4 |

Anh minh chung GPU: `submission/screenshots/01-setup-gpu.png`.

## 3. Ket qua SFT

SFT-mini duoc train tren dataset `bkai-foundation-models/vi-alpaca`. Adapter duoc luu tai:

```text
/content/lab22-light/adapters/sft-mini-light
```

Ket qua:

| Chi so | Gia tri |
|---|---:|
| Trainable params | 4,399,104 |
| Total params | 498,431,872 |
| Trainable ratio | 0.8826% |
| Final SFT loss | 1.3098 |

Anh loss curve: `submission/screenshots/02-sft-loss.png`.

## 4. Preference data va DPO

Preference data duoc lay tu `argilla/ultrafeedback-binarized-preferences-cleaned`, cat 500 mau va format thanh `prompt/chosen/rejected`.

DPO config chinh:

| Tham so | Gia tri |
|---|---:|
| `beta` | 0.1 |
| `learning_rate` | 5e-7 |
| Epoch | 1 |
| Batch | 1 |
| Grad accumulation | 4 |
| Final DPO loss | 0.6922 |

DPO adapter duoc luu tai:

```text
/content/lab22-light/adapters/dpo-light
```

Anh reward curves: `submission/screenshots/03-dpo-reward-curves.png`.

Nhan xet: reward chosen/rejected dao dong quanh 0 va reward gap co bien do nho. Dieu nay cho thay DPO pipeline da chay thanh cong, nhung voi model 0.5B va 500 preference pairs thi alignment signal con yeu. Em khong khang dinh DPO cai thien ro rang; ket qua nen duoc xem la ban chay nhe de kiem tra pipeline.

## 5. So sanh SFT-only va SFT+DPO

Notebook chay 8 prompt gom 4 helpfulness va 4 safety. Anh bang so sanh:

```text
submission/screenshots/04-side-by-side-table.png
```

Ket qua dinh tinh:

- Cac cau helpfulness co cau tra loi nhung con ngan va doi khi sai chi tiet.
- Safety chua on dinh; mot so prompt nguy hiem van chua duoc tu choi dung muc.
- SFT va DPO khac nhau nhe, chua co bang chung DPO thang ro rang.
- Manual rubric hien de `tie` mac dinh, luu o `submission/screenshots/05-manual-rubric.png`.

## 6. Merge va smoke test

De giu ban Light nhe, em khong chay GGUF export. Notebook merge adapter thanh HuggingFace model nhe:

```text
/content/lab22-light/adapters/merged-light
```

Smoke prompt:

```text
Giai thich ngan gon (3 cau) cach thuat toan Bubble sort hoat dong.
```

Model sinh duoc cau tra loi tieng Viet ve Bubble sort. Anh minh chung:

```text
submission/screenshots/06-light-merged-smoke.png
```

## 7. Benchmark

Benchmark nang bang lm-eval duoc skip trong Light mode de giam thoi gian va memory. Notebook van ghi summary:

```text
submission/screenshots/07-light-benchmark-summary.png
```

Thong tin summary:

| Muc | Gia tri |
|---|---|
| Mode | light |
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Num eval prompts | 8 |
| Heavy benchmark | skipped |

## 8. Artifact submission

| File | Noi dung |
|---|---|
| `01-setup-gpu.png` | GPU A100 va runtime |
| `02-sft-loss.png` | SFT loss curve |
| `03-dpo-reward-curves.png` | Chosen/rejected reward va reward gap |
| `04-side-by-side-table.png` | Bang so sanh 8 prompt |
| `05-manual-rubric.png` | Manual rubric |
| `06-light-merged-smoke.png` | Smoke test merged-light |
| `07-light-benchmark-summary.png` | Benchmark summary Light |

## 9. Ket luan

Em da chay that ban Light cua Lab 22 va tao du artifact chinh cho huong nhe. Ket qua DPO loss 0.6922 va SFT loss 1.3098 cho thay pipeline training hoat dong. Tuy nhien model 0.5B va slice du lieu nho nen chat luong output con han che, DPO chua tao cai thien ro rang. Neu co them thoi gian/tai nguyen, buoc tiep theo la tang model len 1.5B/3B, tang preference slice, va chay judge/benchmark day du.
