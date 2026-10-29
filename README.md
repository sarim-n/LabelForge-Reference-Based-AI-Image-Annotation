# LabelForge · AI-Powered Image Auto-Labeling Studio

LabelForge is a zero-shot, human-in-the-loop image annotation workspace built with Streamlit and Python. It offers dual automated detection modes—**Grounding DINO Tiny** for text-prompted detection and a **Training-Free Visual Reference Matcher** using class-neutral Segment Anything Model (SAM) region proposals and frozen DINOv2 feature embeddings—without requiring model training or weight fine-tuning.

> **Model Restrictions & Weight Freezing Guarantee:**
> 1. **No Fine-Tuning or Retraining:** Model weights remain **strictly frozen** during annotation. No backpropagation, fine-tuning, or weight updates ever occur. Checkpoints are downloaded lazily on first use and never at application startup.
> 2. **Class-Neutral Proposals:** For the visual-reference workflow, **no semantic object detectors** (e.g. YOLO, Grounding DINO, OWL-ViT) are used. SAM (`facebook/sam-vit-base`) is employed strictly to generate **class-neutral candidate mask proposals** without semantic awareness.
> 3. **Non-Parametric Exemplar Matching:** Proposed candidate regions are compared with approved reference crops using frozen DINOv2 (`facebook/dinov2-small`) cosine similarity features.
> 4. **No Trained Detector Claim:** SAM proposals and DINOv2 feature matching are **not** a trained category (e.g. car) detector; matching is purely non-parametric visual exemplar retrieval.
> 5. **Separation of Modes:** The text-prompt detector (Grounding DINO) remains completely separate and is never called from the visual-reference workflow.

---

## Phase Status

- [x] **Phase 1: Project Foundation** — Clean architecture, dependency profiles, `.gitignore`, device resolution (`auto`/`CUDA`/`CPU`), onboarding UI, lazy model loading.
- [x] **Phase 2: Automated Detection** — Pretrained Grounding DINO Tiny integration for zero-shot text-prompted bounding box generation, coordinate clamping, low-memory GPU handling, and review integration.
- [x] **Phase 3: Iterative Reference-Based Auto-Labeling (SAM + DINOv2)** — Class-neutral SAM mask proposals, frozen DINOv2 feature comparison, top-K reference scoring, ambiguity margin guard, cache invalidation on annotation changes, and sequential GPU memory management for 4 GB VRAM budgets.
- [x] **Phase 4: Dataset Export, Validation, and Quality Checks** — Pre-export validation, deterministic train/val splits with seed, empty labels for negative background samples, escaped dataset.yaml, export preview, review productivity proxy, and ground-truth Precision/Recall/F1 evaluation.
- [x] **Local Workspace Persistence & Resilient Recovery** — Atomic autosaving to `work/workspace/`, automatic restore on page refresh or server restart, portable project backup ZIPs, zip-slip defense, validation guards, and zero serialization of PyTorch runtime models.

---

## Prerequisites & Hardware Support

- **Operating System:** Windows 10 or Windows 11 (64-bit), Linux, or macOS.
- **Python:** Python 3.10 or 3.11 (Python 3.11 64-bit recommended).
- **RAM:** 16 GB system memory recommended.
- **GPU (Optional):** NVIDIA GPU with 4 GB VRAM (e.g. NVIDIA GeForce RTX 2050/3050 Laptop GPU or higher). Supported via CUDA with automatic CPU fallback.
- **Disk Space:** ~700 MB for lazy Hugging Face model checkpoints.

---

## First-Run Model Download Sizes

All models are downloaded lazily on demand during the first scan; no models are downloaded during application startup:

| Model Component | Identifier / Checkpoint | Download Size | Purpose |
| :--- | :--- | :--- | :--- |
| **Grounding DINO Tiny** | `IDEA-Research/grounding-dino-tiny` | ~170 MB | Zero-shot open-vocabulary text detection |
| **SAM ViT-Base** | `facebook/sam-vit-base` | ~375 MB | Class-neutral candidate region/mask proposals |
| **DINOv2 Small** | `facebook/dinov2-small` | ~85 MB | Frozen visual representation & cosine similarity |

---

## Windows Installation & Run Guide

### 1. Open PowerShell and Create a Virtual Environment

Open PowerShell in this project repository folder:

```powershell
py -3.11 -m venv .venv
```

If PowerShell script execution is restricted on your machine, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the virtual environment:
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

Install lightweight core application dependencies:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

To enable GPU-accelerated model inference (NVIDIA RTX 2050/3050/4050 4 GB+):
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-model.txt
```

For CPU-only machines:
```powershell
pip install -r requirements-model.txt
```

### 3. Start the Application

Launch the Streamlit workspace:
```powershell
streamlit run app.py
```

The application opens at `http://localhost:8501`.

---

## Automated Detection Modes

### Mode 1: Grounding DINO Tiny (Zero-Shot Text Detection)
- **Concept:** Identifies objects matching natural language text prompts (e.g. `car`, `bottle`, `cup`).
- **Isolation:** Operates independently; never called or triggered by the visual reference matcher.
- **Usage:**
  1. In the sidebar under **⚙️ Model & Hardware Setup**, select **Grounding DINO Tiny (zero-shot text detector)**.
  2. Enter class names in Tab **01 · Label & Teach** and click **⚡ Detect Objects with Grounding DINO**.
  3. Approve accurate predictions in Tab **02 · Review Predictions** to add them to your dataset.

### Mode 2: Visual Reference Matcher (SAM + DINOv2)
- **Concept:** Guided predictions driven solely by approved visual reference crops.
- **Workflow Architecture:**
  1. **Region Proposal Generation (SAM):** `facebook/sam-vit-base` automatic mask generation proposes candidate object regions without category labels. Proposals are filtered using predicted IoU quality ($\ge 0.70$), pixel dimensions ($\ge 16\text{px}$), area ratio bounds ($[0.0003, 0.95]$), and NMS deduplication ($\text{IoU} = 0.82$).
  2. **Feature Extraction (Frozen DINOv2):** Candidate regions and approved reference crops are padded to $224 \times 224$ (preserving aspect ratio) and encoded using frozen `facebook/dinov2-small`. L2-normalized CLS token representations produce unit-norm feature embeddings.
  3. **Top-K Similarity Scoring:** For each class, the similarity score for a proposal is computed by averaging its strongest $K$ reference similarities ($K=3$). This ensures classes with more approved references do not automatically dominate smaller classes.
  4. **Ambiguity Margin Guard:** A proposal is assigned to a class only if its similarity meets the threshold and exceeds the second-best class by at least the margin (default: `margin = 0.05`).
  5. **Similarity Scores:** All scores are cosine similarities in $[-1, 1]$, **not calibrated probabilities**, and are clearly labeled as `score` (similarity).
  6. **Spatial De-duplication:** Per-class detection NMS ($\text{IoU} = 0.55$) removes overlapping candidate boxes.

---

## Human-in-the-Loop Review & Reference Bank

- **Unified Reference Bank:** Manual annotations drawn on the canvas and approved automated detections enter the **exact same** class-specific visual reference library (`session_state.refs`).
- **Strict Rejection Exclusion:** Rejected proposals are discarded immediately and **never** enter the reference bank.
- **Embedding Cache & Invalidation:** Reference embeddings are cached in memory using SHA-1 content hashing of the crop pixels. Whenever an annotation is added, approved, or deleted, `invalidate_reference_cache()` clears stale embeddings while keeping loaded models in memory, eliminating redundant recomputation without reloading model weights.
- **Direct Canvas Editing:** Drag, resize, and fine-tune proposal bounding boxes directly on the image canvas in Tab **02 · Review Predictions** before approving.

---

## 4 GB GPU & Low-Memory Optimization Strategy

Designed to run smoothly on laptops with **16 GB RAM** and a **4 GB NVIDIA GPU** (e.g. RTX 2050):
1. **Sequential GPU Offloading:** SAM and DINOv2 never co-occupy VRAM. SAM generates proposals with small point batches (`points_per_batch = 4`), then moves to CPU and calls `torch.cuda.empty_cache()`. DINOv2 is then moved to GPU for small-batch crop encoding (`batch_size = 8`), followed by immediate CPU offloading.
2. **Lazy Loading:** Models are instantiated only when the user clicks "Scan Images" or "Detect Objects".
3. **Workspace Isolation:** Model objects, weights, and PyTorch tensors are **never** serialized to workspace persistence files (`project.json`).
4. **CPU Fallback:** If CUDA runs out of memory or is unavailable, inference falls back gracefully to CPU.

---

## Quantitative Evaluation & Benchmark Results

To evaluate the reference-based auto-labeling workflow without train/reference leakage:
- **Zero Leakage Protocol:** The dataset is partitioned into disjoint reference images and held-out test images. Reference crops are sourced **strictly** from reference images. Held-out test images and their annotations are **never** seen by the reference bank.
- **Greedy IoU Matching:** Matches predictions to ground truth at a standard $\text{IoU} \ge 0.50$ threshold.

### Held-Out Evaluation on Car Practice Set (NVIDIA RTX 2050 Laptop GPU)

| Metric | Measured Value | Description |
| :--- | :--- | :--- |
| **Reference Images** | 8 images (29 crops) | Reference bank guiding predictions |
| **Held-Out Test Images** | 15 images (zero leakage) | Independent evaluation set |
| **Ground Truth Objects** | 29 objects | Target ground truth annotations |
| **Total Predictions** | 27 proposals | Retained after threshold & margin |
| **True Positives (TP)** | 26 | Matches with $\text{IoU} \ge 0.50$ |
| **False Positives (FP)** | 1 | Predictions without matching ground truth |
| **Missed Objects (FN)** | 3 | Ground truth objects not detected |
| **Precision** | **96.30%** (26/27) | $\text{TP} / (\text{TP} + \text{FP})$ |
| **Recall** | **89.66%** (26/29) | $\text{TP} / (\text{TP} + \text{FN})$ |
| **F1 Score** | **92.86%** | $2 \times P \times R / (P + R)$ |
| **Mean Box IoU Quality** | **0.7482** | Average IoU on true positive matches |
| **Average Latency** | **2.19 s / image** | End-to-end SAM proposal + DINOv2 scoring |
| **Throughput** | **0.46 images / s** | On 4 GB NVIDIA RTX 2050 GPU |

### Running the Benchmark

You can reproduce or run the held-out benchmark on your workspace at any time:

```powershell
python evaluate_reference_matcher.py --num-refs 8 --max-test 15 --thresh 0.45 --margin 0.05 --iou 0.50 --device auto --out eval_results.json
```

---

## Local Workspace Persistence & Recovery

- **Atomic Autosave:** Local workspaces autosave to `work/workspace/project.json` using atomic replace operations (`.tmp_*` commit via `os.replace`).
- **Resilient Recovery:** Automatically restores uploaded images, labels, visual references, pending proposals, review counts, and settings upon browser refresh or server restart.
- **Portable Backups:** Download self-contained `.zip` archives with full Zip Slip directory traversal defenses.
- **Privacy:** 100% offline and local. Zero external telemetry or cloud API calls.

---

## Project Layout

```text
ANNO-FLOW/
│
├── app.py                      # Streamlit UI, visual reference gallery, review & YOLO export
├── config.py                   # Model choices, device resolution, and hardware detection
├── evaluate_reference_matcher.py # Held-out evaluation benchmark (Precision, Recall, F1, IoU, throughput)
├── persistence.py              # Atomic local autosave, restore, schema validation, and backup ZIPs
├── test_persistence.py         # Unit tests for persistence, atomic writes & Zip Slip defenses
├── vision.py                   # GroundingDinoDetector, ReferenceMatcher (SAM+DINOv2), IoU, crop utils
│
├── requirements.txt            # Core UI dependencies (Streamlit, Canvas, Pillow, NumPy)
├── requirements-model.txt      # Model inference dependencies (PyTorch, Transformers, OpenCV)
├── .gitignore                  # Ignores venvs, checkpoints (*.pt, *.safetensors), datasets, work/workspace/
└── README.md                   # Complete architectural guide, benchmark results & setup manual
```
