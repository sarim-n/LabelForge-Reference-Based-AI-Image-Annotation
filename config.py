"""Configuration layer for model choices, compute device selection, and phase options.

Phase 4: Dataset Export, Validation, and Quality Checks.
Keeps imports lightweight so the Streamlit application starts instantly without
loading heavy machine learning packages or downloading model weights on startup.
"""

from typing import Any, Dict, Tuple
import os

WORKSPACE_DIR: str = os.getenv("LABELFORGE_WORKSPACE_DIR", "work/workspace")
WORKSPACE_SCHEMA_VERSION: int = 1

MODEL_GROUNDING_DINO = "Text-Prompt Object Detector"
MODEL_CLIP_REFERENCE = "Visual Reference Matcher"
MODEL_CHOICES: Tuple[str, ...] = (MODEL_CLIP_REFERENCE, MODEL_GROUNDING_DINO)

DEVICE_AUTO = "auto"
DEVICE_CUDA = "CUDA"
DEVICE_CPU = "CPU"
DEVICE_CHOICES: Tuple[str, ...] = (DEVICE_AUTO, DEVICE_CUDA, DEVICE_CPU)

PROPOSAL_OPENCV = "OpenCV (Fast / Lightweight)"
PROPOSAL_SAM = "Deep Visual Masks (High Detail)"
PROPOSAL_METHOD_CHOICES: Tuple[str, ...] = (PROPOSAL_OPENCV, PROPOSAL_SAM)


# Metadata describing model options
MODEL_INFO: Dict[str, Dict[str, str]] = {
    MODEL_GROUNDING_DINO: {
        "status": "Active (Phase 2)",
        "type": "Zero-shot open-vocabulary object detector",
        "description": (
            "Detects objects via text prompts (e.g. 'bottle', 'car') without fine-tuning weights. "
            "Operates in pure zero-shot inference mode."
        ),
        "weights": "Frozen pretrained vision checkpoint. Zero fine-tuning.",
        "download_timing": "Lazy on-demand download during first detection run; never at startup.",
    },
    MODEL_CLIP_REFERENCE: {
        "status": "Active (Phase 3)",
        "type": "Training-free reference matcher with class-neutral region proposals",
        "description": (
            "Proposes candidate regions and compares them against approved annotation crops using deep visual similarity. "
            "Every approved box expands the shared visual reference bank."
        ),
        "weights": "Frozen visual region proposals and exemplar visual features. No detector head or retraining.",
        "download_timing": "Vision checkpoints download lazily on the first visual-reference scan.",
    },
}


def resolve_device(preference: str = DEVICE_AUTO, torch_module: Any = None) -> Tuple[str | None, str]:
    """Detect compute device (CUDA GPU or CPU) with friendly status and CPU fallback.

    Parameters
    ----------
    preference : str
        User preference: "auto", "CUDA", or "CPU".
    torch_module : module, optional
        Imported PyTorch module. If None, an import is attempted dynamically
        without making PyTorch a hard dependency for app startup.

    Returns
    -------
    tuple of (device_str, status_message)
        device_str is 'cuda', 'cpu', or None (if torch is missing).
    """
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore
        except ImportError:
            if preference == DEVICE_CPU:
                return "cpu", "CPU selected. Install PyTorch to enable automated predictions."
            return None, "PyTorch is not installed. Workspace and manual annotations remain fully operational."

    try:
        cuda_available = bool(torch_module.cuda.is_available())
        device_name = torch_module.cuda.get_device_name(0) if cuda_available else ""
    except Exception as exc:
        if preference != DEVICE_CUDA:
            return "cpu", f"CUDA check encountered an error ({exc}); falling back to CPU."
        return None, f"Could not verify CUDA ({exc}). Select CPU or check your PyTorch installation."

    if preference == DEVICE_CPU:
        return "cpu", "CPU selected. Automated inference will run on the processor."

    if preference == DEVICE_CUDA:
        if cuda_available:
            return "cuda", f"CUDA available ({device_name}). GPU acceleration active."
        return "cpu", "CUDA unavailable on this system; falling back to CPU. Install CUDA PyTorch and drivers for GPU use."

    # preference == "auto"
    if cuda_available:
        return "cuda", f"CUDA detected ({device_name}); automatic mode will use GPU acceleration."
    return "cpu", "CUDA unavailable; automatic mode defaulted to CPU."


def get_hardware_acceleration_tip() -> str | None:
    """Detect if an NVIDIA GPU is physically present while PyTorch is running in CPU-only mode."""
    try:
        import torch
        if torch.cuda.is_available():
            return None
    except Exception:
        pass

    try:
        import shutil
        import subprocess

        if shutil.which("nvidia-smi"):
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True,
                timeout=2,
            )
            first_line = out.strip().splitlines()[0] if out.strip() else ""
            if first_line:
                return (
                    f"NVIDIA GPU detected on system (**{first_line}**), but PyTorch was installed without CUDA acceleration. "
                    "Inference is currently running on the CPU (~25s/image). "
                    "Install CUDA PyTorch for ~0.7s/image (35x faster): "
                    "`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --upgrade`"
                )
    except Exception:
        pass
    return None

