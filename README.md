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
python -m pip install setuptools wheel
python -m pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/detectron2.git@02b5c4e295e990042a714712c21dc79b731e8833"
```

## Data Preparation

Follow the data preparation instructions in
[DSEC-Detection](https://github.com/uzh-rpg/dsec-det).
Include its [DSEC-extra](https://github.com/uzh-rpg/dsec-det#dsec-extra)
sequences and [remapped images](https://github.com/uzh-rpg/dsec-det#remapped-images).

Place the prepared dataset in `datasets/DSEC/`, or link your existing directory:

```bash
mkdir -p datasets
ln -s /path/to/dsec datasets/DSEC
```

Run the commands below from the repository root. All scripts use this data
directory by default; to change it, edit `path` in
[config/dataset/dsec.yaml](config/dataset/dsec.yaml) once.

Then generate `events_2x.h5` for every sequence with the script included here:

```bash
for split in train test; do
  for sequence in datasets/DSEC/"$split"/*; do
    python scripts/downsample_events.py \
      "$sequence/events/left/events.h5" \
      "$sequence/events/left/events_2x.h5"
  done
done
```

Your final data directory should look like this:

```text
datasets/DSEC/
├── train/
│   ├── <training_sequence>/
│   ├── zurich_city_16_a/       # validation
│   ├── ...                    # zurich_city_17_a through zurich_city_20_a
│   └── zurich_city_21_a/       # validation
└── test/
    └── <test_sequence>/
```

Every sequence, including the validation sequences, has the same layout:

```text
<sequence>/
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

All values are COCO AP (%) on the DSEC-Det test split.

### Standard Global

| Checkpoint | AP | AP50 | AP75 | APS | APM | APL |
|---|---:|---:|---:|---:|---:|---:|
| `flexevent_2class_global.pth` | 59.52 | 79.23 | 69.78 | 54.56 | 68.83 | 83.21 |
| `flexevent_2class_balanced.pth` | 58.16 | 80.40 | 66.71 | 52.22 | 67.12 | 84.60 |
| `flexevent_8class_global.pth` | 53.40 | 75.31 | 62.89 | 40.04 | 61.47 | 78.19 |
| `flexevent_8class_balanced.pth` | 51.76 | 75.96 | 61.17 | 40.39 | 59.12 | 79.14 |

### Interframe

| Evaluation | 2-class Global | 2-class Balanced | 8-class Global | 8-class Balanced |
|---|---:|---:|---:|---:|
| Empty event | 34.21 | **37.40** | 27.83 | **33.99** |
| 180 Hz | 46.94 | **52.64** | 41.84 | **45.57** |
| 90 Hz | 55.60 | **56.77** | 47.69 | **50.82** |
| 60 Hz | 57.65 | **58.40** | 49.70 | **52.66** |
| 45 Hz | 59.10 | **59.32** | 52.00 | **53.18** |
| 36 Hz | **60.16** | 59.66 | 52.98 | **53.86** |
| 30 Hz | **60.47** | 59.62 | **54.28** | 54.19 |
| 20 Hz | **60.15** | 59.06 | **54.63** | 53.38 |

## Evaluation

### Global

```bash
python test_global.py \
  dataset.class_mode=2class \
  checkpoint=checkpoints/flexevent_2class_global.pth
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
  dataset.class_mode=2class \
  checkpoint=checkpoints/flexevent_2class_balanced.pth \
  validation.interframe=true
```

### Visualization

`visualize.py` runs one test sequence and renders RGB, events, ground truth and
predictions together. Use an image extension for one frame or a video extension
for a sequence:

```bash
python visualize.py \
  dataset.class_mode=2class \
  checkpoint=checkpoints/flexevent_2class_balanced.pth \
  visualization.sequence=thun_01_a \
  visualization.output=demo.mp4
```

## Training

DSEC-Det provides frame-aligned annotations at 20 Hz. Training follows the two
components of FlexEvent: **FlexFuse** learns to combine event and RGB information using these annotations, and
**FlexTune** extends supervision to intermediate timestamps for detection at
higher frequencies.

For the strongest performance at the standard **20 Hz** setting, we train
FlexFuse with GT labels, following Section 3.2 of the paper. Event views with
different temporal durations are fused with RGB features and supervised at the
original annotated timestamps. This produces `flexevent_2class_global.pth` and
`flexevent_8class_global.pth`.

For more balanced performance from **20 Hz to 180 Hz**, we additionally use
FlexTune (Section 3.3). Higher-frequency detection uses shorter event windows
and requires predictions between annotated frames. We first train a Teacher on
shorter event windows ending at the existing GT timestamps, then use it to
generate and temporally filter pseudo-labels at intermediate timestamps. A
Student is initialized from the Teacher and fine-tuned with both GT and
pseudo-labels. This produces `flexevent_2class_balanced.pth` and
`flexevent_8class_balanced.pth`, which trade some standard-frequency accuracy
for stronger high-frequency performance.

### 1. Pretrained initialization

The event branch is initialized from RVT-B trained on the Prophesee 1Mpx
detection dataset. The RGB branch uses an ImageNet
ResNet-50. Download the [RVT-B weights](https://download.ifi.uzh.ch/rpg/RVT/checkpoints/1mpx/rvt-b.ckpt)
and the ImageNet ResNet-50
[R-50.pkl](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/torchvision/R-50.pkl).
Place them in `pretrained/rvt-b.ckpt` and `pretrained/R-50.pkl`.
Training loads these weights by default.

### 2. Standard Global training

This trains FlexFuse with GT labels:

```bash
python train.py recipe=2class_global
```

Use `recipe=8class_global` for the multi-class model.

### 3. FlexTune Balanced training

#### Train the Teacher

Following the paper's low-frequency sparse training stage, we train on the
final part of each event interval that ends at an annotated timestamp. The
main event window is 25 ms, and the labels remain at
their original 20 Hz timestamps while the event windows are shorter.

```bash
python train.py recipe=2class_teacher
```

Use `recipe=8class_teacher` for the multi-class Teacher.

#### Generate pseudo-labels

Run the Teacher in both temporal directions, then filter its predictions:

```bash
for direction in forward reverse; do
  python generate_pseudo_labels.py \
    dataset.class_mode=2class \
    checkpoint=/path/to/teacher.ckpt direction=$direction \
    output_dir=pseudo_predictions
done

python scripts/build_pseudo_labels.py \
  --predictions-root pseudo_predictions \
  --output-root pseudo_labels --class-mode 2class
```

For the multi-class model, use `8class` in generation and label construction,
and add `--classwise` to the label-construction command. Pseudo-labels are
thresholded before union NMS and track filtering; short rejected tracks are
pruned.

#### Train the Student

```bash
python train.py recipe=2class_balanced \
  dataset.train.pseudo_labels_root=pseudo_labels \
  training.teacher_checkpoint=/path/to/teacher.ckpt
```

Use `recipe=8class_balanced` for the multi-class Student. Both Students train on
GT at annotated timestamps and pseudo-labels at intermediate timestamps, starting
from the Teacher weights with a new optimizer and schedule.

All evaluation scripts accept either a released `.pth` file or a Lightning
`.ckpt` written during training.

## Acknowledgements

This code builds on
[RVT](https://github.com/uzh-rpg/RVT),
[DSEC-Detection](https://github.com/uzh-rpg/dsec-det), and
[DAGR](https://github.com/uzh-rpg/dagr).
