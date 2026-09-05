import os
import sys
from pathlib import Path
from typing import Optional

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.backends import cuda, cudnn

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.modifier import dynamically_modify_inference_config
from data.dsec_utils.class_config import get_dsec_class_spec
from data.dsec_utils.dsec_data import DSEC
from data.dsec_utils.dsec_det.dataset import DSECDet
from data.dsec_utils.dsec_det.io import yaml_file_to_dict
from data.utils.types import DataType
from models.detection.yolox.postprocess import postprocess
from modules.utils.fetch import fetch_model_module

cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".avi"}
CLASS_COLORS = (
    (255, 128, 0),
    (0, 200, 255),
    (0, 200, 0),
    (200, 0, 200),
    (0, 100, 255),
    (255, 0, 128),
    (128, 255, 0),
)


def _load_weights(module, checkpoint: Path) -> None:
    module.load_state_dict(torch.load(str(checkpoint), map_location="cpu"))


def _event_image(event_repr: torch.Tensor) -> np.ndarray:
    event_repr = event_repr.detach().cpu().float()
    if event_repr.ndim != 3 or event_repr.shape[0] != 20:
        raise ValueError(f"Expected a [20,H,W] event tensor, got {tuple(event_repr.shape)}")
    negative = event_repr[:10].sum(dim=0).numpy()
    positive = event_repr[10:].sum(dim=0).numpy()
    peak = float(np.percentile(np.maximum(negative, positive), 99))
    scale = max(peak, 1.0)
    negative = np.clip(negative / scale, 0, 1)
    positive = np.clip(positive / scale, 0, 1)

    height, width = negative.shape
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    strength = np.maximum(negative, positive)
    image[..., 1] = np.round(255 * (1 - strength)).astype(np.uint8)
    image[..., 0] = np.round(255 * (1 - positive)).astype(np.uint8)
    image[..., 2] = np.round(255 * (1 - negative)).astype(np.uint8)
    return image


def _draw_box(image: np.ndarray, box, class_id: int, text: str) -> None:
    height, width = image.shape[:2]
    x0, y0, x1, y1 = (int(round(float(value))) for value in box)
    x0, x1 = np.clip((x0, x1), 0, width - 1)
    y0, y1 = np.clip((y0, y1), 0, height - 1)
    color = CLASS_COLORS[class_id % len(CLASS_COLORS)]
    cv2.rectangle(image, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
    text_size, baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
    )
    text_y = max(y0, text_size[1] + baseline + 2)
    cv2.rectangle(
        image,
        (x0, text_y - text_size[1] - baseline - 2),
        (min(width - 1, x0 + text_size[0] + 4), text_y + 1),
        color,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x0 + 2, text_y - baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def _draw_ground_truth(
        image: np.ndarray, labels, class_names) -> np.ndarray:
    output = image.copy()
    if labels is None:
        return output
    values = labels.object_labels.detach().cpu().numpy()
    for value in values:
        x, y, width, height = value[1:5]
        class_id = int(value[5])
        _draw_box(
            output,
            (x, y, x + width, y + height),
            class_id,
            class_names[class_id],
        )
    return output


def _draw_predictions(
        image: np.ndarray, predictions: Optional[torch.Tensor], class_names
        ) -> np.ndarray:
    output = image.copy()
    if predictions is None:
        return output
    for value in predictions.detach().cpu().numpy():
        class_id = int(value[6])
        score = float(value[4] * value[5])
        _draw_box(
            output,
            value[:4],
            class_id,
            f"{class_names[class_id]} {score:.2f}",
        )
    return output


def _panel(image: np.ndarray, title: str) -> np.ndarray:
    header = np.full((24, image.shape[1], 3), 28, dtype=np.uint8)
    cv2.putText(
        header,
        title,
        (7, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return np.concatenate((header, image), axis=0)


def _compose_frame(
        rgb: np.ndarray,
        event_repr: torch.Tensor,
        labels,
        predictions: Optional[torch.Tensor],
        class_names,
        sequence: str,
        frame_index: int,
        timestamp: int,
        confidence_threshold: float,
        ) -> np.ndarray:
    top = np.concatenate(
        (_panel(rgb, "RGB"), _panel(_event_image(event_repr), "Events")),
        axis=1,
    )
    bottom = np.concatenate(
        (
            _panel(_draw_ground_truth(rgb, labels, class_names), "Ground truth"),
            _panel(
                _draw_predictions(rgb, predictions, class_names),
                f"Prediction (score >= {confidence_threshold:.2f})",
            ),
        ),
        axis=1,
    )
    header = np.full((30, top.shape[1], 3), 15, dtype=np.uint8)
    cv2.putText(
        header,
        f"{sequence} | frame {frame_index} | t={int(timestamp)} us",
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return np.concatenate((header, top, bottom), axis=0)


def _build_sequence(config: DictConfig, sequence: str) -> DSEC:
    split_file = PROJECT_ROOT / "data/dsec_utils/dsec_split.yaml"
    split_config = yaml_file_to_dict(split_file)
    if sequence not in split_config["test"]:
        choices = ", ".join(split_config["test"])
        raise ValueError(f"Unknown test sequence {sequence!r}. Choose from: {choices}")
    base = DSECDet(
        root=Path(config.dataset.path).expanduser(),
        split="test",
        sync="back",
        split_config={"test": [sequence]},
    )
    dual = config.dataset.flexfuse.dual_frequency
    return DSEC(
        base,
        sequence_name=sequence,
        sequence_length=int(config.dataset.sequence_length),
        min_bbox_diag=float(config.dataset.min_bbox_diag),
        min_bbox_height=float(config.dataset.min_bbox_height),
        num_us=float(config.dataset.num_us),
        dual_frequency=bool(dual.enable),
        high_window_ratio=float(dual.high_window_ratio),
        high_window_crop=str(dual.inference_crop),
        high_window_seed=int(dual.inference_seed),
        class_mode=str(config.dataset.class_mode),
    )


@hydra.main(config_path="config", config_name="visualize", version_base="1.2")
def main(config: DictConfig) -> None:
    dynamically_modify_inference_config(config)
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    gpu = config.hardware.gpus
    if not isinstance(gpu, int) or gpu < 0:
        raise ValueError("hardware.gpus must select one CUDA device")
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)

    split_config = yaml_file_to_dict(
        PROJECT_ROOT / "data/dsec_utils/dsec_split.yaml"
    )
    sequence = config.visualization.sequence
    sequence = split_config["test"][0] if sequence is None else str(sequence)
    dataset = _build_sequence(config, sequence)

    module = fetch_model_module(config=config)
    _load_weights(module, Path(config.checkpoint).expanduser())
    module.to(device).eval()
    model = module.mdl
    class_names = get_dsec_class_spec(
        str(config.dataset.class_mode)
    ).model_class_names

    output_path = Path(config.visualization.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix not in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
        raise ValueError("visualization.output must end in png/jpg/bmp/mp4/avi")
    write_video = suffix in VIDEO_SUFFIXES

    start_frame = int(config.visualization.start_frame)
    frame_stride = int(config.visualization.frame_stride)
    max_frames = int(config.visualization.max_frames)
    fps = float(config.visualization.fps)
    confidence_threshold = float(config.visualization.confidence_threshold)
    if start_frame < 0 or frame_stride < 1 or max_frames < 1 or fps <= 0:
        raise ValueError("Invalid visualization frame or fps setting")
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be in [0, 1]")

    video_writer = None
    previous_states = None
    frame_index = 0
    written = 0
    amp_enabled = str(config.precision) in {"16", "16-mixed"}

    with torch.inference_mode():
        for chunk_index in range(len(dataset)):
            sample = dataset[chunk_index]
            labels_sequence = sample[DataType.OBJLABELS_SEQ]
            images = sample[DataType.IMAGE]
            padded = sample[DataType.IS_PADDED_MASK]
            timestamps = sample[DataType.TIMESTAMP]
            dual_frequency = DataType.EV_REPR_B in sample
            events_a = sample[
                DataType.EV_REPR_A if dual_frequency else DataType.EV_REPR
            ]
            events_b = sample[DataType.EV_REPR_B] if dual_frequency else None

            for time_index, is_padded in enumerate(padded):
                if is_padded:
                    continue
                event_a_cpu = events_a[time_index]
                event_a = module.input_padder.pad_tensor_ev_repr(
                    event_a_cpu.unsqueeze(0).to(device=device, dtype=torch.float32)
                )
                event_b = None
                if dual_frequency:
                    event_b = module.input_padder.pad_tensor_ev_repr(
                        events_b[time_index].unsqueeze(0).to(
                            device=device, dtype=torch.float32
                        )
                    )
                image = images[time_index].unsqueeze(0).to(device=device)
                with torch.autocast(
                        device_type="cuda", dtype=torch.float16,
                        enabled=amp_enabled):
                    features, previous_states = model.forward_backbone(
                        x=event_a,
                        image=image,
                        previous_states=previous_states,
                        x_b=event_b,
                    )
                    predictions = model.forward_detect(features)
                    predictions = postprocess(
                        prediction=predictions,
                        num_classes=len(class_names),
                        confidence_threshold=confidence_threshold,
                        nms_threshold=float(config.model.postprocess.nms_threshold),
                    )[0]

                selected = (
                    frame_index >= start_frame
                    and (frame_index - start_frame) % frame_stride == 0
                )
                if selected:
                    rgb = images[time_index].permute(1, 2, 0).cpu().numpy()
                    rendered = _compose_frame(
                        rgb=rgb,
                        event_repr=event_a_cpu,
                        labels=labels_sequence[time_index],
                        predictions=predictions,
                        class_names=class_names,
                        sequence=sequence,
                        frame_index=frame_index,
                        timestamp=timestamps[time_index],
                        confidence_threshold=confidence_threshold,
                    )
                    if write_video:
                        if video_writer is None:
                            codec = "mp4v" if suffix == ".mp4" else "MJPG"
                            video_writer = cv2.VideoWriter(
                                str(output_path),
                                cv2.VideoWriter_fourcc(*codec),
                                fps,
                                (rendered.shape[1], rendered.shape[0]),
                            )
                            if not video_writer.isOpened():
                                raise RuntimeError(f"Could not open video output {output_path}")
                        video_writer.write(rendered)
                    else:
                        if not cv2.imwrite(str(output_path), rendered):
                            raise RuntimeError(f"Could not write image {output_path}")
                    written += 1

                frame_index += 1
                if written >= (max_frames if write_video else 1):
                    break
            if written >= (max_frames if write_video else 1):
                break

    if video_writer is not None:
        video_writer.release()
    if written == 0:
        raise ValueError(
            f"start_frame={start_frame} is beyond the {frame_index} evaluated frames"
        )
    print(f"Saved {written} frame(s) to {output_path}")


def _disable_hydra_outputs() -> None:
    defaults = (
        "hydra.run.dir=.",
        "hydra.output_subdir=null",
        "hydra/job_logging=disabled",
        "hydra/hydra_logging=disabled",
    )
    missing = []
    for value in defaults:
        key = value.split("=", 1)[0]
        if not any(arg.split("=", 1)[0] == key for arg in sys.argv[1:]):
            missing.append(value)
    first_option = next(
        (index for index, arg in enumerate(sys.argv[1:], start=1)
         if arg.startswith("--")),
        len(sys.argv),
    )
    sys.argv[first_option:first_option] = missing


if __name__ == "__main__":
    _disable_hydra_outputs()
    main()
