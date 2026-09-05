<p align="center">
  <h2 align="center">FlexEvent: Towards Flexible Event-Frame Object Detection at Varying Operational Frequencies</h2>
  <p align="center">
    <a href="https://dylanorange.github.io">Dongyue Lu</a><sup>1,2</sup>&nbsp;&nbsp;
    <a href="https://ldkong.com">Lingdong Kong</a><sup>1</sup>&nbsp;&nbsp;
    <a href="https://www.comp.nus.edu.sg/~leegh/">Gim Hee Lee</a><sup>1</sup>&nbsp;&nbsp;
    <a href="https://scholar.google.at/citations?user=mbMNjekAAAAJ&hl=en">Camille Simon Chane</a><sup>3</sup>&nbsp;&nbsp;
    <a href="https://www.comp.nus.edu.sg/~ooiwt/">Wei Tsang Ooi</a><sup>1,2</sup>
  </p>
  <p align="center">
    <sup>1</sup>National University of Singapore&nbsp;&nbsp;
    <sup>2</sup>IPAL, CNRS IRL 2955, Singapore<br>
    <sup>3</sup>ETIS UMR 8051, CY Cergy Paris University, ENSEA, CNRS, France
  </p>
  <p align="center">
    <a href="https://arxiv.org/abs/2412.06708"><img src="https://img.shields.io/badge/Paper-arXiv-lightblue"></a>
    <a href="https://flexevent.github.io"><img src="https://img.shields.io/badge/Project-Page-blue"></a>
    <a href="https://huggingface.co/datasets/dylanorange/FlexEvent"><img src="https://img.shields.io/badge/Models-Hugging%20Face-yellow"></a>
  </p>
</p>

<p align="center">
  <img src="docs/webpage2.gif" alt="FlexEvent detections at varying frequencies" width="900">
</p>

## About

FlexEvent is an event-frame object detector designed to work across varying
operational frequencies. It combines the temporal resolution of event cameras
with the semantic detail of RGB frames, allowing the same detector to operate
from standard frame rates up to 180 Hz.

Our main contributions are:

- **FlexEvent**, a flexible event-frame detection framework for real-world
  sensing at varying operational frequencies.
- **FlexFuse**, an adaptive fusion module that combines recurrent event
  features with multi-scale RGB features.
- **FlexTune**, a frequency-adaptive fine-tuning method that uses
  frequency-adjusted labels to improve temporal generalization.

## Installation


```bash
conda create -n flexevent python=3.9 -y
conda activate flexevent

python -m pip install \
  torch==2.1.2 torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

## Data Preparation

Follow the data preparation instructions in
[DSEC-Detection](https://github.com/uzh-rpg/dsec-det).

Then generate `events_2x.h5` for every sequence with the script included here:

```bash
export DSEC_ROOT=/path/to/dsec
for split in train test; do
  for sequence in "$DSEC_ROOT/$split"/*; do
    python scripts/downsample_events.py \
      "$sequence/events/left/events.h5" \
      "$sequence/events/left/events_2x.h5"
  done
done
```

Your final data directory should look like this:

```text
DSEC_ROOT/
├── train/<sequence>/
└── test/<sequence>/
    ├── images/
    │   ├── timestamps.txt
    │   └── left/distorted/*.png
    ├── events/left/events_2x.h5
    └── object_detections/left/tracks.npy
```

We provide two label settings:

- `2class`: `car` and `pedestrian`. The original car, bus, and truck
  categories are merged into `car`.
- `8class`: the native DSEC categories. Since the `train` category has no
  annotations in the DSEC-Det split, the released model evaluates the seven
  non-empty categories.

## Checkpoints and Results

Download the released weights from
[Hugging Face](https://huggingface.co/datasets/dylanorange/FlexEvent/tree/main/checkpoints)
and place them in `checkpoints/`.

For each label setting, we provide two variants:

- **Global** is a FlexFuse model trained only with ground-truth labels. It
  generally performs best at the standard 20 Hz operating frequency.
- **Balanced** adds FlexTune and is further trained with pseudo-labels at
  intermediate timestamps. It performs better at high frequencies and is more
  consistent across the full frequency range.

All values below are COCO AP@[0.50:0.95].

### Standard Global

| Checkpoint | AP | AP50 | AP75 | APS | APM | APL |
|---|---:|---:|---:|---:|---:|---:|
| `flexevent_2class_global.pth` | 0.595150 | 0.792337 | 0.697793 | 0.545560 | 0.688259 | 0.832088 |
| `flexevent_2class_balanced.pth` | 0.581633 | 0.803953 | 0.667099 | 0.522205 | 0.671182 | 0.845990 |
| `flexevent_8class_global.pth` | 0.533950 | 0.753108 | 0.628857 | 0.400416 | 0.614706 | 0.781865 |
| `flexevent_8class_balanced.pth` | 0.517571 | 0.759601 | 0.611672 | 0.403929 | 0.591239 | 0.791363 |

### Interframe

| Evaluation | 2-class Global | 2-class Balanced | 8-class Global | 8-class Balanced |
|---|---:|---:|---:|---:|
| Standard Global | **0.595150** | 0.581633 | **0.533950** | 0.517571 |
| Empty event | 0.342114 | **0.373957** | 0.278341 | **0.339851** |
| 180 Hz | 0.469367 | **0.526417** | 0.418439 | **0.455698** |
| 90 Hz | 0.555963 | **0.567673** | 0.476913 | **0.508179** |
| 60 Hz | 0.576454 | **0.584011** | 0.496976 | **0.526610** |
| 45 Hz | 0.590995 | **0.593196** | 0.519990 | **0.531825** |
| 36 Hz | **0.601558** | 0.596619 | 0.529823 | **0.538618** |
| 30 Hz | **0.604662** | 0.596181 | **0.542758** | 0.541876 |
| 20 Hz | **0.601496** | 0.590635 | **0.546277** | 0.533758 |

## Evaluation

### Global

```bash
python test_global.py \
  dataset=dsec \
  dataset.path=/path/to/dsec \
  dataset.class_mode=2class \
  checkpoint=checkpoints/flexevent_2class_global.pth \
  hardware.gpus=0
```

Use `dataset.class_mode=8class` with an 8-class checkpoint.

### Interframe

Our interframe time sampling and interpolated labels follow
[DAGR](https://github.com/uzh-rpg/dagr). It reports the empty-event point and
nine event rates from 180 Hz to 20 Hz. Intermediate annotations are generated
online from the official DSEC-Det tracks by matching track IDs and interpolating
the bounding boxes between two annotated frames.
No additional label-generation step is required.

```bash
python test_interframe.py \
  dataset=dsec \
  dataset.path=/path/to/dsec \
  dataset.class_mode=2class \
  checkpoint=checkpoints/flexevent_2class_balanced.pth \
  hardware.gpus=0 \
  validation.interframe=true
```

### Visualization

`visualize.py` runs one test sequence and renders RGB, events, ground truth and
predictions together. Use an image extension for one frame or a video extension
for a sequence:

```bash
python visualize.py \
  dataset=dsec \
  dataset.path=/path/to/dsec \
  dataset.class_mode=2class \
  checkpoint=checkpoints/flexevent_2class_balanced.pth \
  hardware.gpus=0 \
  visualization.sequence=thun_01_a \
  visualization.output=demo.mp4
```

## Acknowledgements

This code builds on
[RVT](https://github.com/uzh-rpg/RVT),
[DSEC-Detection](https://github.com/uzh-rpg/dsec-det), and
[DAGR](https://github.com/uzh-rpg/dagr).

## Training

Training code will be released soon.
