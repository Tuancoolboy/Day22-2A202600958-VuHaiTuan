# Reflection — Lab 22 DPO Alignment Light Run

**Ten:** Vu Hai Tuan  
**MSSV:** 2A202600958  
**Ngay:** 27/06/2026  
**Notebook da chay:** `Lab22_DPO_Light.ipynb`  
**Che do:** LIGHT / LOW_MEM

---

## 1. Muc tieu va ly do chon ban Light

Muc tieu cua Lab 22 la di qua pipeline alignment tu SFT sang preference learning: tao SFT-mini adapter, chuan bi du lieu preference, train DPO adapter, so sanh SFT-only voi SFT+DPO, va tao bang/anh minh chung de nop bai. Ban goc cua lab co huong T4/BigGPU dung model 3B/7B va co cac buoc nang nhu GGUF, lm-eval benchmark. Trong lan chay nay em uu tien mot ban nhe, it loi, chay duoc nhanh trong moi truong co RAM he thong thap, nen em dung notebook Light.

Ban Light su dung `Qwen/Qwen2.5-0.5B-Instruct`, LoRA rank 8, `MAX_LEN=384`, `MAX_PROMPT_LEN=192`, SFT slice 300 mau va preference slice 500 mau. Em tat gradient checkpointing vi luc dau training bi loi `element 0 of tensors does not require grad`; sau khi tat checkpointing, SFT va DPO deu chay thanh cong. Doi lai, ket qua khong the so sanh truc tiep voi BigGPU 7B trong README, nhung phu hop voi muc tieu "chay that, nhe, co artifact va screenshot that".

## 2. Moi truong chay

Notebook duoc chay tren GPU A100:

| Thanh phan | Gia tri |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB |
| CUDA | 13.0 |
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| dtype | `torch.bfloat16` |
| SFT slice | 300 mau |
| Preference slice | 500 cap chosen/rejected |
| Max length | 384 |
| Max prompt length | 192 |
| Adapter SFT | `/content/lab22-light/adapters/sft-mini-light` |
| Adapter DPO | `/content/lab22-light/adapters/dpo-light` |
| Merged model | `/content/lab22-light/adapters/merged-light` |

Anh minh chung moi truong nam o `submission/screenshots/01-setup-gpu.png`.

## 3. Ket qua SFT-mini

SFT-mini duoc train tren 300 mau tu `bkai-foundation-models/vi-alpaca`. Muc dich cua buoc nay khong phai tao model manh, ma tao mot adapter ban dau de DPO co policy SFT lam diem xuat phat.

Ket qua:

| Chi so | Gia tri |
|---|---:|
| Trainable parameters | 4,399,104 |
| All parameters | 498,431,872 |
| Trainable ratio | 0.8826% |
| Final SFT loss | 1.3098 |

Loss curve duoc luu o `submission/screenshots/02-sft-loss.png`. Vi slice rat nho, em xem SFT-mini nay la checkpoint thuc nghiem de kiem tra pipeline, khong xem la model tieng Viet chat chat luong cao.

## 4. Preference data va DPO training

Du lieu preference duoc lay tu `argilla/ultrafeedback-binarized-preferences-cleaned`, cat 500 mau dau tien. Notebook format thanh cac cot `prompt`, `chosen`, `rejected` theo chat template cua Qwen.

DPO duoc train voi:

| Tham so | Gia tri |
|---|---:|
| `beta` | 0.1 |
| `learning_rate` | 5e-7 |
| `num_train_epochs` | 1 |
| `per_device_train_batch_size` | 1 |
| `gradient_accumulation_steps` | 4 |
| `loss_type` | sigmoid |
| Final DPO loss | 0.6922 |

Reward curves duoc luu o `submission/screenshots/03-dpo-reward-curves.png`. Ket qua reward trong ban Light dao dong quanh 0, reward gap co luc duong va co luc am, bien do nho. Dieu nay cho thay DPO da chay va co tin hieu log reward, nhung voi model 0.5B + 500 preference pairs thi alignment signal con yeu. Em khong ket luan model "tot hon ro rang"; em chi ket luan pipeline DPO da chay thanh cong va co artifact that.

Day cung la bai hoc quan trong cua DPO: reward gap khong nen duoc doc mot cach may moc. Neu gap tang nhung chosen reward khong tang on dinh, co the do likelihood displacement hoac noise tu slice nho. Trong ket qua cua em, duong chosen/rejected deu nho va dao dong, nen can than trong dien giai.

## 5. So sanh SFT-only voi SFT+DPO

Notebook sinh bang side-by-side voi 8 prompt, gom 4 prompt helpfulness va 4 prompt safety. Anh bang nam o `submission/screenshots/04-side-by-side-table.png`.

Quan sat nhanh:

| Nhom prompt | Nhan xet |
|---|---|
| Helpfulness | SFT va DPO deu tra loi duoc nhung con ngan, doi khi giai thich sai hoac thieu. Vi model 0.5B va SFT slice nho nen kha nang reasoning/cong thuc con han che. |
| Safety | Mot so cau nguy hiem van chua refuse tot, dac biet prompt ve hoa chat no va mua ruou. Cau ve tin nhan khung bo co xu huong refuse tot hon. |
| Khac biet SFT/DPO | Khac biet co nhung khong lon; DPO chua tao cai thien ro rang tren tat ca prompt. |

Vi khong dung API judge, notebook de che do manual rubric. Anh `submission/screenshots/05-manual-rubric.png` ghi trang thai manual/tie de em tu danh gia lai khi nop. Em de `tie` mac dinh vi voi ban Light, chat luong hai adapter gan nhau va chua du co so de cham DPO thang ro rang.

## 6. Merge va smoke test

De giu notebook nhe, em khong export GGUF Q4_K_M. Thay vao do, notebook merge adapter DPO vao model HuggingFace nhe va smoke test prompt:

> Giai thich ngan gon (3 cau) cach thuat toan Bubble sort hoat dong.

Model merged-light da sinh duoc cau tra loi tieng Viet mach lac ve Bubble sort. Anh minh chung nam o `submission/screenshots/06-light-merged-smoke.png`.

Day la thay the nhe cho GGUF smoke test trong README. Neu can nop dung day du NB5 theo rubric goc, can chay them GGUF export; tuy nhien trong muc tieu cua em, uu tien la ban nhe chay on dinh va khong loi.

## 7. Benchmark va trade-off

Notebook Light bo qua lm-eval nang nhu IFEval, GSM8K, MMLU va AlpacaEval-lite. Tom tat benchmark nhe duoc luu o `submission/screenshots/07-light-benchmark-summary.png`:

| Muc | Gia tri |
|---|---|
| Mode | light |
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Num eval prompts | 8 |
| Heavy benchmark | skipped |

Em chap nhan trade-off nay vi muc tieu la chay that trong cau hinh nhe. Benchmark day du se can nhieu thoi gian va tai nguyen hon, trong khi notebook Light tap trung vao viec chung minh pipeline SFT -> DPO -> eval -> merge smoke.

## 8. Dieu em hoc duoc

Dieu quan trong nhat em hoc duoc la alignment pipeline khong chi la goi trainer. Mot thay doi nho ve memory/gradient nhu `gradient_checkpointing=True` co the lam PEFT/LoRA bi mat gradient va loi backward. Sau khi tat gradient checkpointing, model 0.5B chay on dinh hon ma van vua memory.

Em cung thay ro rang rang DPO can doc ket qua rat can than. Loss 0.6922 gan voi muc log loss ban dau cua preference classification, reward gap dao dong nho, va qualitative output chua cai thien manh. Neu muon ket qua tot hon, huong tiep theo nen la:

- Tang preference slice tu 500 len 1000-2000 neu RAM cho phep.
- Dung model 1.5B hoac 3B thay vi 0.5B.
- Chay beta sweep voi `beta` 0.05, 0.1, 0.5.
- Cai thien preference data tieng Viet thay vi chi dung UltraFeedback tieng Anh.
- Chay judge/API hoac manual rubric nghiem tuc thay vi de tie mac dinh.

## 9. Ket luan

Em da chay thanh cong ban Light cua Lab 22 tren A100, tao du SFT adapter, preference data, DPO adapter, side-by-side eval, merged-light model va cac screenshot submission. Ket qua khong manh nhu BigGPU 7B, nhung la mot pipeline that, nhe va co the tai lap. Reflection nay vi vay khong con ghi placeholder; tat ca so lieu chinh nhu SFT loss 1.3098, DPO loss 0.6922, 8 prompt eval va merged-light smoke deu lay tu notebook da chay.
