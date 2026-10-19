"""
Detect NSFW — one ComfyUI node.

- IMAGE input: Freepik NSFWDetector (same check as NSFW Check).
- STRING input: Qwen3-VL prompt guard (same prompt and verdict rules).
"""

import json
import os
import re
import threading
from enum import Enum

import torch
from PIL import Image

import comfy.model_management
import comfy.sd
import folder_paths
from nodes import interrupt_processing
from server import PromptServer


# ---------------------------------------------------------------------------
# Freepik image detector (copied from comfyui-nsfw-freepik)
# ---------------------------------------------------------------------------

HF_MODEL_ID = "Freepik/nsfw_image_detector"
LOCAL_MODEL_DIR = os.path.join(folder_paths.models_dir, "nsfw_freepik")


def _model_source() -> str:
    if os.path.isfile(os.path.join(LOCAL_MODEL_DIR, "config.json")):
        return LOCAL_MODEL_DIR
    print(f"[Detect NSFW] {LOCAL_MODEL_DIR} not found, downloading {HF_MODEL_ID} from Hugging Face")
    return HF_MODEL_ID


class NSFWContentError(Exception):
    """Custom exception for NSFW content detection."""

    def __init__(self, prediction: str, confidence: float, threshold: float):
        self.error_type = "nsfw_content_detected"
        self.prediction = prediction
        self.confidence = confidence
        self.threshold = threshold
        self.nsfw_percent = confidence * 100.0
        message = (
            "NSFW content detected - workflow interrupted "
            f"(prediction: {prediction}, confidence: {confidence:.2%}, "
            f"NSFW: {self.nsfw_percent:.2f}%, threshold: {threshold:.2%})"
        )
        super().__init__(message)

    def to_dict(self):
        return {
            "error": {
                "type": self.error_type,
                "message": str(self),
                "details": {
                    "prediction": self.prediction,
                    "confidence": self.confidence,
                    "nsfw_percent": self.nsfw_percent,
                    "threshold": self.threshold,
                },
            }
        }


# ---------------------------------------------------------------------------
# Qwen text guard (copied from comfyui_qwen_nsfw_guard, prompt path only)
# ---------------------------------------------------------------------------

DEFAULT_CLIP = "qwen3vl_8b_int8_convrot.safetensors"
_cached_clip = None
_cached_path = None
_load_lock = threading.Lock()

_PROMPT_POLICY = """Evaluate this prompt for workplace safety. Prompt must be in English (otherwise VERDICT: NOT_ENGLISH).
Safety policy: NSFW = nudity, sexual requests, erotic content, exposed body parts; clothed people/everyday scenes = SAFE.
Reply with exactly two lines:
VERDICT: SAFE, VERDICT: NSFW, or VERDICT: NOT_ENGLISH
REASON: brief explanation.
"""


class NSFWDetectedError(RuntimeError):
    pass


class UnsupportedLanguageError(ValueError):
    pass


def _load_clip(clip_name):
    global _cached_clip, _cached_path
    path = folder_paths.get_full_path("text_encoders", clip_name)
    if path is None:
        path = folder_paths.get_full_path("clip", clip_name)
    if path is None:
        raise FileNotFoundError(f"Qwen NSFW Guard: text encoder not found: {clip_name}")
    with _load_lock:
        if _cached_clip is None or _cached_path != path:
            loaded = comfy.sd.load_clip(ckpt_paths=[path])
            _cached_clip, _cached_path = loaded, path
        return _cached_clip


def _check(clip, query, source, image=None):
    tokens = clip.tokenize(query) if image is None else clip.tokenize(query, images=[image])
    generated = clip.generate(tokens, do_sample=False, max_length=32, temperature=0.0)
    response = clip.tokenizer.decode(generated).strip()
    verdicts = re.findall(r"^\s*VERDICT:\s*(NSFW|SAFE|NOT[ _]ENGLISH|UNSUPPORTED[ _]LANGUAGE)\b", response, re.MULTILINE | re.IGNORECASE)
    if any(re.match(r"^(NOT[ _]ENGLISH|UNSUPPORTED[ _]LANGUAGE)$", v, re.IGNORECASE) for v in verdicts):
        comfy.model_management.interrupt_current_processing(True)
        raise UnsupportedLanguageError("language not supported, only support english at this time")
    if any(verdict.upper() == "NSFW" for verdict in verdicts):
        comfy.model_management.interrupt_current_processing(True)
        raise NSFWDetectedError(f"Qwen NSFW Guard blocked {source}: {response}")
    if len(verdicts) != 1 or verdicts[0].upper() != "SAFE":
        raise RuntimeError(f"Qwen NSFW Guard could not classify {source}: {response!r}")
    return {"source": source, "verdict": "SAFE", "response": response}


def _moderate_prompt(clip_name, clip=None, prompt=None):
    if prompt is None or not str(prompt).strip():
        raise ValueError("Qwen NSFW Guard requires a prompt.")
    clip = clip if clip is not None else _load_clip(clip_name)
    checks = [
        _check(
            clip,
            _PROMPT_POLICY + "\nClassify this prompt (JSON string):\n" + json.dumps(prompt),
            "prompt",
        )
    ]
    return json.dumps({"verdict": "SAFE", "checks": checks}, ensure_ascii=False)


def _log_error(exc):
    print(f"[Detect NSFW] {exc}", flush=True)


class DetectNSFW:
    """Image -> Freepik. String -> Qwen prompt guard. Errors are printed to the log."""

    _detector = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "display": "slider",
                        "tooltip": (
                            "Chặn workflow nếu NSFWLevel.LOW >= giá trị này. "
                            "Mặc định 0.5: low >= 0.5 sẽ bị chặn."
                        ),
                    },
                ),
                "clip_name": ("STRING", {"default": DEFAULT_CLIP}),
            },
            "optional": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "clip": ("CLIP",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("image", "prompt", "report", "nsfw_percent")
    FUNCTION = "check"
    CATEGORY = "safety"
    OUTPUT_NODE = True

    def _get_detector(self):
        from nsfw_image_detector import NSFWDetector

        if DetectNSFW._detector is not None:
            return DetectNSFW._detector

        model_name = _model_source()

        # Prefer requested config (bfloat16 on cuda). Fall back if unsupported.
        try:
            DetectNSFW._detector = NSFWDetector(model_name=model_name, dtype=torch.bfloat16, device="cuda")
        except Exception as e1:
            print(f"[Detect NSFW] Failed init cuda+bfloat16, falling back: {e1}")
            try:
                DetectNSFW._detector = NSFWDetector(model_name=model_name, dtype=torch.float16, device="cuda")
            except Exception as e2:
                print(f"[Detect NSFW] Failed init cuda+float16, falling back to CPU: {e2}")
                DetectNSFW._detector = NSFWDetector(model_name=model_name, dtype=torch.float32, device="cpu")

        return DetectNSFW._detector

    def _normalize_prob_key(self, k) -> str:
        # predict_proba uses enum keys like NSFWLevel.LOW; str(k) is not "low".
        if isinstance(k, Enum):
            return str(k.value)
        return str(k)

    def _floatify_prob_value(self, v):
        try:
            if hasattr(v, "item") and callable(getattr(v, "item", None)):
                return float(v.item())
            return float(v)
        except Exception:
            return None

    def _coerce_probabilities(self, probs):
        # Normalize outputs from NSFWDetector.predict_proba into dict[str, float|None].
        def dict_to_out(d: dict) -> dict:
            out = {}
            for k, v in d.items():
                nk = self._normalize_prob_key(k)
                out[nk] = self._floatify_prob_value(v)
            return out

        if isinstance(probs, dict):
            return dict_to_out(probs)

        if isinstance(probs, (list, tuple)) and probs:
            for item in probs:
                if isinstance(item, dict):
                    return dict_to_out(item)

        return probs

    def _extract_low_score(self, probs_dict: dict) -> float:
        # Freepik-style probabilities: high/medium/low/neutral (keys after coercion).
        v = probs_dict.get("low")
        try:
            return float(v) if v is not None else 0.0
        except Exception:
            return 0.0

    def _extract_low_score_any(self, probs_raw) -> float:
        coerced = self._coerce_probabilities(probs_raw)
        if isinstance(coerced, dict):
            return self._extract_low_score(coerced)
        return 0.0

    def _check_image(self, image: torch.Tensor, threshold: float):
        detector = self._get_detector()
        batch_size = image.shape[0]
        results = []
        nsfw_detected = False
        max_low_score = 0.0

        for i in range(batch_size):
            # VAE Qwen Image 2.1 trả RGBA (4 kênh); ép 4 kênh thành "RGB" làm detector thấy ảnh nhiễu
            img_tensor = image[i][..., :3].float().clamp(0.0, 1.0)
            img_np = (img_tensor.cpu().numpy() * 255).round().astype("uint8")
            pil_image = Image.fromarray(img_np, mode="RGB")

            probs_raw = detector.predict_proba(pil_image)
            probs = self._coerce_probabilities(probs_raw)

            low_score = self._extract_low_score_any(probs_raw)
            try:
                low_score = float(low_score)
            except Exception:
                low_score = 0.0

            # Optional boolean classification (kept for reference in results)
            try:
                is_nsfw_flag = bool(detector.is_nsfw(pil_image, "low"))
            except Exception as e:
                print(f"[Detect NSFW] is_nsfw failed for i{i}: {e}")
                is_nsfw_flag = False

            if low_score > max_low_score:
                max_low_score = low_score

            # blocked khi low >= threshold (vd threshold=0.5 → low >= 0.5)
            blocked = low_score >= threshold
            if blocked:
                nsfw_detected = True

            nsfw_percent = low_score * 100.0
            print(
                f"[Detect NSFW] image[{i}] NSFW: {nsfw_percent:.2f}% "
                f"(threshold: {threshold:.2%}, blocked: {blocked})",
                flush=True,
            )

            results.append(
                {
                    "index": i,
                    "low_score": low_score,
                    "nsfw_percent": nsfw_percent,
                    "threshold": threshold,
                    "blocked": blocked,
                    "is_nsfw_flag": is_nsfw_flag,
                    "probabilities": probs,
                }
            )

        nsfw_percent = max_low_score * 100.0
        if nsfw_detected:
            error = NSFWContentError(
                prediction="nsfw",
                confidence=max_low_score,
                threshold=threshold,
            )
            _log_error(error)

            PromptServer.instance.send_sync(
                "nsfw_guard.content_blocked",
                {
                    "type": "nsfw_content_detected",
                    "prediction": "nsfw",
                    "confidence": max_low_score,
                    "nsfw_percent": nsfw_percent,
                    "threshold": threshold,
                    "results": results,
                    "message": str(error),
                },
            )

            interrupt_processing(True)
            raise Exception(json.dumps(error.to_dict()))

        report = {
            "verdict": "SAFE",
            "nsfw_percent": nsfw_percent,
            "threshold": threshold,
            "results": results,
        }
        return nsfw_percent, report

    def _check_prompt(self, prompt, clip_name, clip):
        try:
            report = _moderate_prompt(clip_name, clip=clip, prompt=prompt)
        except (NSFWDetectedError, UnsupportedLanguageError, RuntimeError, FileNotFoundError, ValueError) as exc:
            _log_error(exc)
            raise
        return report

    def check(self, threshold=0.5, clip_name=DEFAULT_CLIP, image=None, prompt=None, clip=None):
        has_image = image is not None
        has_prompt = isinstance(prompt, str) and prompt.strip() != ""
        if not has_image and not has_prompt:
            exc = ValueError("Detect NSFW requires an image or a text prompt.")
            _log_error(exc)
            raise exc

        nsfw_percent = 0.0
        report_obj = {}

        if has_image:
            nsfw_percent, image_report = self._check_image(image, threshold)
            report_obj["image"] = image_report

        prompt_out = prompt if has_prompt else ""
        if has_prompt:
            text_report = self._check_prompt(prompt, clip_name, clip)
            report_obj["prompt"] = json.loads(text_report)

        report = json.dumps(report_obj, ensure_ascii=False)
        return {"ui": {"text": [report]}, "result": (image, prompt_out, report, nsfw_percent)}


NODE_CLASS_MAPPINGS = {
    "DetectNSFW": DetectNSFW,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DetectNSFW": "Detect NSFW",
}
