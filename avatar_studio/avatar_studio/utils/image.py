"""Image IO + face alignment (FFHQ-style 1024 crop using dlib landmarks)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import PIL.Image
import scipy.ndimage
import torch
import torchvision.transforms as T


# ----- tensor <-> PIL -----

def pil_to_tensor(img: PIL.Image.Image, size: int = 1024) -> torch.Tensor:
    """RGB PIL → tensor in [-1, 1], shape (1, 3, H, W)."""
    tfm = T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.LANCZOS),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return tfm(img.convert("RGB")).unsqueeze(0)


def tensor_to_pil(t: torch.Tensor) -> PIL.Image.Image:
    """Tensor in [-1, 1], shape (C, H, W) or (1, C, H, W) → PIL."""
    if t.dim() == 4:
        t = t[0]
    t = (t.clamp(-1, 1) + 1) / 2
    arr = (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return PIL.Image.fromarray(arr)


def save_image(t: torch.Tensor, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(t).save(path)


# ----- FFHQ-style alignment (dlib 68-landmarks) -----

_DLIB_DETECTOR = None
_DLIB_PREDICTOR = None


def _ensure_dlib(landmarks_path: str):
    global _DLIB_DETECTOR, _DLIB_PREDICTOR
    if _DLIB_PREDICTOR is None:
        import dlib
        _DLIB_DETECTOR = dlib.get_frontal_face_detector()
        _DLIB_PREDICTOR = dlib.shape_predictor(landmarks_path)


def get_landmarks(img: PIL.Image.Image, landmarks_path: str) -> np.ndarray:
    _ensure_dlib(landmarks_path)
    import dlib
    arr = np.array(img.convert("RGB"))
    dets = _DLIB_DETECTOR(arr, 1)
    if len(dets) == 0:
        raise RuntimeError("No face detected.")
    shape = _DLIB_PREDICTOR(arr, dets[0])
    return np.array([[p.x, p.y] for p in shape.parts()])


def align_face(
    img_path: str | Path,
    landmarks_path: str,
    output_size: int = 1024,
    transform_size: int = 4096,
    enable_padding: bool = True,
) -> PIL.Image.Image:
    """FFHQ alignment — same recipe as the original StyleGAN preprocessing."""
    img = PIL.Image.open(img_path).convert("RGB")
    lm = get_landmarks(img, landmarks_path)

    lm_eye_left      = lm[36:42]
    lm_eye_right     = lm[42:48]
    lm_mouth_outer   = lm[48:60]

    eye_left     = np.mean(lm_eye_left,  axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_avg    = (lm_mouth_outer[0] + lm_mouth_outer[6]) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)),
                 int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.LANCZOS)
        quad /= shrink
        qsize /= shrink

    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))),  int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0),
            min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
           int(np.ceil(max(quad[:, 0]))),  int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0),
           max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), "reflect")
        h, w, _ = img.shape
        y_, x_, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x_) / pad[0],
                                           np.float32(w - 1 - x_) / pad[2]),
                          1.0 - np.minimum(np.float32(y_) / pad[1],
                                           np.float32(h - 1 - y_) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), "RGB")
        quad += pad[:2]

    img = img.transform((transform_size, transform_size), PIL.Image.QUAD,
                        (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.LANCZOS)
    return img
