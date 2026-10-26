# Pet Breed Classifier — Oxford-IIIT Pet (37 კლასი)

Computer Vision-ის end-to-end პროექტი: ძაღლისა და კატის ჯიშის ამოცნობა სურათიდან,
მონაცემების მომზადებიდან გაწვრთნილ მოდელამდე, ოპტიმიზაციამდე და მოქმედ API-მდე.

| მეტრიკა | მნიშვნელობა |
|---|---|
| Test accuracy | **88.55%** (3 669 სურათი) |
| Macro F1 | **0.884** |
| საბოლოო მოდელი | MobileNetV2 fine-tuning (2.2M პარამეტრი) |
| Inference | ONNX Runtime, CPU — **9.0 ms/სურათი** (PyTorch: 25.2 ms) |

---

## პროექტის სტრუქტურა

```
pet-breed-classifier/
├── src/                     # ვარჯიში, შეფასება, ექსპორტი
│   ├── config.py            # ყველა პარამეტრი ერთ ადგილას (seed, batch, lr, MEAN/STD)
│   ├── data.py              # transforms, train/val/test გაყოფა, DataLoader-ები
│   ├── models.py            # SimpleCNN (baseline), ResNet18, MobileNetV2
│   ├── train.py             # training loop და ექსპერიმენტების გაშვება
│   ├── evaluate.py          # test მეტრიკები, confusion matrix, შეცდომების ანალიზი
│   ├── gradcam.py           # ვიზუალიზაცია: რას უყურებს მოდელი
│   └── export_onnx.py       # ONNX ექსპორტი + CPU ბენჩმარკი + სისწორის შემოწმება
├── app/                     # FastAPI სერვისი (torch-ის გარეშე)
│   ├── main.py              # endpoint-ები
│   ├── predictor.py         # preprocessing + ONNX inference
│   ├── classes.json         # 37 ჯიში, მოდელის ინდექსების რიგით
│   └── model_meta.json      # input size, MEAN/STD — API იგივე preprocessing-ს იმეორებს
├── demo/gradio_app.py       # ვიზუალური დემო სურათის ატვირთვით
├── tests/test_api.py        # API-ის ტესტები
├── notebooks/               # ექსპერიმენტების notebook (Colab)
├── models/                  # .onnx / .pth (repo-ში არ ინახება — იხ. ქვემოთ)
├── reports/                 # ტექნიკური ანგარიში, confusion matrix, Grad-CAM
├── requirements.txt         # მხოლოდ inference/API
└── requirements-train.txt   # + torch, torchvision, sklearn...
```

**რატომ ორი requirements ფაილი:** სერვისი ONNX Runtime-ზე მუშაობს და `torch` არ სჭირდება.
ეს deployment-ის ზომას ასობით მეგაბაიტით ამცირებს.

---

## გაშვება

### 1. API (მოდელის მოსინჯვა)

```bash
pip install -r requirements.txt
# მოათავსე mobilenet_pet.onnx საქაღალდეში models/
uvicorn app.main:app --reload
```

გახსენი **http://127.0.0.1:8000/docs** → `/predict` → **Try it out** → ატვირთე სურათი.

| Endpoint | მეთოდი | აღწერა |
|---|---|---|
| `/health` | GET | სერვისისა და მოდელის მდგომარეობა |
| `/classes` | GET | 37 ჯიშის სია |
| `/predict` | POST | სურათი → top-k ჯიში confidence-ით |

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@my_cat.jpg" -F "top_k=3"
```

```json
{
  "filename": "my_cat.jpg",
  "predictions": [
    {"breed": "Abyssinian", "confidence": 0.9712},
    {"breed": "Bengal",     "confidence": 0.0151},
    {"breed": "Egyptian Mau","confidence": 0.0042}
  ]
}
```

### 2. Gradio დემო

```bash
pip install -r requirements-train.txt
python demo/gradio_app.py
```

### 3. მოდელის ხელახლა გაწვრთნა

```bash
pip install -r requirements-train.txt
python -m src.train --experiment all        # ყველა ექსპერიმენტი (GPU-ზე ~10 წთ)
python -m src.evaluate --checkpoint models/E_mobilenet_v2_best.pth --arch mobilenet
python -m src.gradcam
python -m src.export_onnx                   # -> models/mobilenet_pet.onnx + app/classes.json
```

მონაცემები (~800 MB) პირველ გაშვებაზე ავტომატურად ჩამოიტვირთება `data/`-ში.
GPU არ არის სავალდებულო, მაგრამ CPU-ზე ვარჯიში მნიშვნელოვნად ნელია.
Colab-ის უფასო T4-ზე ერთი ექსპერიმენტი ≈ 2 წუთია.

### 4. ტესტები

```bash
pytest -q
```

---

## მეთოდოლოგია

**მონაცემები.** Oxford-IIIT Pet: 3 680 `trainval` + 3 669 `test`.
`trainval` იყოფა 80/20-ად ფიქსირებული seed-ით (42), ანუ გაყოფა ყოველ გაშვებაზე იგივეა.
`test` ვარჯიშისას საერთოდ არ გამოიყენება და მასზე მხოლოდ ერთი, წინასწარ არჩეული მოდელი ფასდება.

**Augmentation.** `RandomResizedCrop`, `RandomHorizontalFlip`, `ColorJitter` მხოლოდ train-ზე.
Val/test დეტერმინისტულია, რომ გაზომვა განმეორებადი იყოს.

**Normalize.** ImageNet-ის MEAN/STD, რადგან pretrained backbone-ები ამ სტატისტიკაზე გაიწვრთნენ.
იგივე პარამეტრები `app/model_meta.json`-ში გადადის, რომ API-მ ზუსტად იგივე preprocessing გაიმეოროს.

**Checkpoint.** ინახება საუკეთესო val accuracy-ის ეპოქა (early stopping-ის მარტივი ფორმა).

---

## ექსპერიმენტები

| # | მიდგომა | პარამეტრები | Val accuracy |
|---|---|---|---|
| A | SimpleCNN ნულიდან (15 ეპოქა) | 0.4M | 24.0% |
| B | ResNet18, გაყინული backbone | 11.2M (ვარჯიშობს 0.02M) | 87.2% |
| C | ResNet18, სრული fine-tuning | 11.2M | 90.5% |
| D | C + AdamW(wd=1e-2) + label smoothing + cosine LR | 11.2M | 90.9% |
| E | **MobileNetV2, fine-tuning** | **2.2M** | **90.9%** |

**დასკვნები:**

1. Pretrained წონები 24% → 87%-ზე გადადის მხოლოდ ბოლო ფენის ვარჯიშითაც. მცირე მონაცემებზე
   (2 944 სურათი, ~80 თითო კლასზე) transfer learning გადამწყვეტია.
2. Backbone-ის ადაპტაცია ამატებს ~3 პუნქტს (B → C).
3. C, D და E ერთმანეთისგან პრაქტიკულად განურჩეველია: val = 736 სურათი, ანუ 0.4 პუნქტი ≈ 3 სურათი.
   სტატისტიკურად მნიშვნელოვან სხვაობას ვერ ვამტკიცებთ.
4. ამიტომ საბოლოო მოდელად **MobileNetV2** ავირჩიეთ: იგივე სიზუსტე, 5-ჯერ ნაკლები პარამეტრი.

**ოპტიმიზაციამდე და შემდეგ:**

| | პარამეტრები | ფაილის ზომა | CPU ms/სურათი |
|---|---|---|---|
| ResNet18 (PyTorch) | 11.2M | ~45 MB | — |
| MobileNetV2 (PyTorch) | 2.2M | 9.3 MB | 25.2 ms |
| **MobileNetV2 (ONNX Runtime)** | 2.2M | 9.26 MB | **9.0 ms** (2.8×) |

გამოსავლების მაქსიმალური სხვაობა PyTorch-სა და ONNX-ს შორის: **8.9e-07**, ანუ სიზუსტე არ შეცვლილა.
ONNX-მა **ზომა არ შეამცირა** — მხოლოდ CPU inference აჩქარა.

---

## შეფასება და შეცდომების ანალიზი

Test accuracy **88.55%**, macro F1 **0.884** (val-ზე 90.9%).
2.4 პუნქტიანი სხვაობა მოსალოდნელია: checkpoint val-ით ავირჩიეთ, ამიტომ val ოდნავ ოპტიმისტურია.

შეცდომები შემთხვევითი არ არის და სამ ჯგუფად იყოფა:

| ჯგუფი | აღრევა | მიზეზი |
|---|---|---|
| ბულ-ტიპის ძაღლები | Pit Bull → Staffordshire (23), → American Bulldog (12) | თითქმის იდენტური აგებულება |
| გრძელბეწვიანი კატები | Birman ↔ Ragdoll (19 / 13) | ერთნაირი წერტილოვანი შეფერილობა |
| მოკლებეწვიანი კატები | Russian Blue ↔ British Shorthair (10 / 10) | ერთი ფერი, მომრგვალო სახე |

American Pit Bull Terrier-ს precision 0.864, recall კი 0.510 — მოდელი იშვიათად ამბობს „Pit Bull",
სამაგიეროდ Staffordshire-ს precision 0.602-ია, ანუ ის სხვა ძაღლებსაც მიეწერება.

**Grad-CAM** ადასტურებს, რომ სწორი პროგნოზებისას მოდელი ცხოველის სახესა და სხეულს უყურებს და არა ფონს.
შეცდომები ჯგუფდება არატიპიურ პოზებზე (მძინარე, ზურგზე მწოლიარე ცხოველი), სადაც სახე არ ჩანს;
ასეთ შემთხვევებში confidence დაბალია, რაც დემოში ფილტრად გამოდგება.

დეტალები: [`reports/technical_report.md`](reports/technical_report.md),
`reports/confusion_matrix.png`, `reports/gradcam.png`.

---

## გაწვრთნილი მოდელი

`models/mobilenet_pet.onnx` (≈9 MB) და `.pth` checkpoint-ები repo-ში არ ინახება.
ორი გზა:

1. **ჩამოტვირთვა:** იხ. release/ბმული (ან ატვირთე ფაილი `models/`-ში ხელით).
2. **ხელახლა მიღება:** `python -m src.train --experiment mobilenet && python -m src.export_onnx`
   — უფასო Colab T4-ზე ჯამში ~5 წუთი.

---

## გამოყენებული ბიბლიოთეკები

PyTorch, torchvision (ვარჯიში) · scikit-learn (მეტრიკები) · matplotlib (ვიზუალიზაცია) ·
ONNX, ONNX Runtime (ოპტიმიზაცია და inference) · FastAPI, Uvicorn, Pillow, NumPy (სერვისი) ·
Gradio (დემო) · pytest (ტესტები).

ზუსტი ვერსიები: `requirements.txt`, `requirements-train.txt`.
