# Dataset Sources — SkinTypeAI YOLO11 acne-detection migration

This file documents every dataset **considered** and the one **selected** for training
the YOLO11 acne-lesion detector, with licensing and provenance. It is part of the
project's academic-integrity record.

SkinTypeAI does **not** own, host, or redistribute any of these datasets. Raw dataset
files live only under `datasets/external/` and `datasets/skin_detect/`, both of which
are git-ignored (see repo `.gitignore`).

---

## SELECTED — ACNE04 (Wu et al., ICCV 2019)

| Field | Value |
|---|---|
| Dataset name | **ACNE04** |
| Original authors | Xiaoping Wu, Ni Wen, Jie Liang, Yu-Kun Lai, Dongyu She, Ming-Ming Cheng, Jufeng Yang |
| Original paper | *Joint Acne Image Grading and Counting via Label Distribution Learning*, IEEE International Conference on Computer Vision (ICCV), 2019. <http://xiaopingwu.cn/assets/paper/iccv2019_ldl.pdf> |
| Original source / repository | <https://github.com/xpwu95/LDL> |
| Distribution | Authors' Google Drive (`Classification.tar` = images, `Detection.tar` = VOC annotations) and Baidu Pan. |
| Original purpose | Acne severity grading (Hayashi criterion, 4 levels) **and** lesion counting. Ships lesion bounding boxes for the counting task. |
| License / usage restriction (verbatim from the repo README) | *"ATTN: This work is free for academic usage. For other purposes, please contact Xiaoping Wu (xpwu95@163.com)."* |
| Usage in this project | **Academic / research use only**, consistent with the above. This is a final-year research project. The dataset is **not redistributed**; only locally derived YOLO-format label files are produced, and those are git-ignored. |
| Date accessed | 2026-08-31 |
| Images | 1457 facial photographs, JPEG, original resolution (e.g. 3112×3456, 4912×3264; varies per image). |
| Annotation type | Pascal VOC XML, one file per image, `<bndbox><xmin><ymin><xmax><ymax></bndbox>`. |
| Bounding boxes | **18,983** lesion boxes, drawn by professional dermatologists (per the paper). min 1 / max 65 / mean 13.0 / median 7 per image. |
| Classes | **1** — VOC object name `fore` (foreground = acne lesion). |
| Official splits | 5-fold, `ImageSets/Main/NNEW_trainval_{0..4}.txt` (1165) + `NNEW_test_{0..4}.txt` (292). 2-way (trainval / test) only. |

### Audit performed by this project (2026-08-31, programmatic — see `training/yolo/` scripts)

```
Total images:                     1457
Total annotation files:           1457
Matched image/annotation pairs:   1457  (100%, by filename stem)
Missing annotations / images:     0 / 0
Corrupt / unreadable images:      0
Invalid XML:                      0
Invalid boxes (xmin>=xmax|ymin>=ymax):  0
Out-of-bounds boxes:              0
Image-dim vs XML-<size> mismatches:     0   (images are original resolution)
Total bounding boxes:             18,983
Classes found:                    {'fore': 18983}
Exact-duplicate images (md5):     38 groups of 2  ->  1419 unique image contents
Official fold-0 split leakage:    12 duplicate pairs straddle trainval<->test
```

### Class mapping applied

```
VOC  "fore"  (acne lesion)   ->   YOLO class 0 = "acne"
```

### Transformations performed by this project

1. **Deduplication** — the 38 byte-identical duplicate images are collapsed to one
   representative each (1457 files → 1419 unique images). The dropped filenames are
   listed in `datasets/README_YOLO.md`.
2. **VOC → YOLO conversion** (`training/yolo/voc_to_yolo.py`) — each `<bndbox>` in
   absolute pixel coords is converted to normalized `class cx cy w h` using the
   **image's own** width/height (verified equal to the XML `<size>`), then validated
   (`0<=cx,cy<=1`, `0<w,h<=1`).
3. **Re-split** — the official 2-way split is **not** used verbatim because (a) it is
   trainval/test only, not train/val/test, and (b) it contains the 12 leakage pairs
   above. A deterministic split (`seed=42`, stratified by Hayashi grade,
   70 / 20 / 10) is produced at the unique-image level. See `datasets/README_YOLO.md`.
4. No pixel edits, no re-encoding, no annotation edits, no synthetic images or boxes.

---

## REJECTED candidates

| Dataset | Where | Why rejected |
|---|---|---|
| `ManuelHettich/acne04` (+ clones `tanmay22`, `Damini13`, `4sleep`, `timsep`) | HuggingFace | Image-level severity labels only (`metadata.jsonl` = `{"file_name","prompt":"photo of a person with acneN"}`). **No bounding boxes.** Also anisotropically resized to 1024². Would require fabricating boxes — forbidden. (Kept as an emergency fallback source of *images* only, since box alignment was verified, but not needed now that `Classification.tar` is available.) |
| `SyedAliJafri/acne-detection-dataset` | HuggingFace | Has YOLO boxes, but **no license stated**, no attribution, unattributed Roboflow re-upload, tiny low-res images (e.g. 294×222), undocumented multi-class scheme. Fails the "licensing must be clear" bar. |
| `makhresearch/skin-lesion-segmentation-classification` | HuggingFace | HAM10000-derived **dermoscopy** close-ups (skin-cancer lesions: MEL/NV/BCC/…). Wrong domain (not facial acne / dark spots). MIT-licensed but irrelevant. |
| `AliRHo/pigmentation-melasma-dark-spots-…` | Roboflow Universe | **Classification only** (819 images, classes normal/dark spots/melasma/…). No boxes. |
| Roboflow Universe acne detection sets (`andrei-dore-5lz05/acne04`, `acne-vulgaris-detection/acne04-detection`, `acne-severity/*`, `dermafind/acne`, several CC BY 4.0) | Roboflow Universe | Real boxes and acceptable licenses, but **programmatic download needs a Roboflow API key** not available in this environment. Most are themselves re-exports of ACNE04, which we now have at source. |
| `Nexdata/Human_Facial_Skin_Defects_Data`, `UniDataPro/facial-skin-condition-dataset` | HuggingFace | Commercial data-vendor sample sets ("contact us" licensing). Terms unclear → excluded. |

---

## `dark_spot` / hyperpigmentation detection — NO USABLE DATASET

An extensive search (Roboflow Universe, Kaggle, HuggingFace, academic repositories,
GitHub) found **no publicly available facial dark-spot / hyperpigmentation dataset with
genuine bounding-box annotations and clear licensing.** Everything available is
image-level classification, wrong-domain dermoscopy, or unclear licensing.

**Decision:** the detector ships with **`acne` only** (class 0). `dark_spot` is a
documented missing class, to be added only if a legitimate box-annotated dataset is
later obtained. No fabricated dark-spot boxes.
