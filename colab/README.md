# Training the jersey number ResNet

`run_models.py` now reads jersey numbers with PARSeq by default (see the main README); the ResNet
is used only with `--number-reader resnet`. Everything for training it is here. The model
itself is defined in `jersey_net.py` at the repo root, since `run_models.py` and
`export_engines.py` use it too.

| File | Purpose |
|---|---|
| `train_jersey.ipynb` | Colab notebook: runs the two scripts below on a Colab GPU |
| `make_jersey_labels.py` | turns a YOLO digit dataset (one box per digit, classes `0`-`9`) into whole-number labels (`labels.csv`) |
| `train_jersey.py` | trains the ResNet and saves the best epoch |

## On Colab

1. In Google Drive, make a folder `Hockey-Vision` containing `jersey_net.py` (repo root),
   `make_jersey_labels.py` and `train_jersey.py` (this folder), and your dataset zipped as
   `jersey_dataset.zip` (with its `data.yaml`).
2. In Colab, *File → Upload notebook* → `train_jersey.ipynb`, then *Runtime → Change runtime type
   → GPU*.
3. Check the options in the first code cell and *Runtime → Run all*.

The notebook checks the GPU, unzips the dataset onto Colab's own disk, makes the labels, cuts
every image to the exact torso crop the model sees (Colab has only ~2 CPU cores, too few to do
that every epoch), shows a sample of crops to check, and trains. The best epoch is saved to Drive
as it improves, so a disconnect doesn't lose it. The free tier disconnects after ~90 minutes
idle, so keep the tab open.

## Roboflow datasets

- Export as **YOLOv8** (any YOLO txt format works). The notebook can download it directly: fill
  in the `ROBOFLOW_*` options and add your API key as a Colab secret named `ROBOFLOW_API_KEY`.
- Export without **flip** augmentation (it mirrors the digits), ideally without any augmentation,
  since training augments on its own. **Stretch** resizing is fine (the crop is resized to a fixed
  size anyway); **Fit (black edges)** letterboxing is not, as it moves where the torso is.
- Images must be **player crops** (one player each), not full broadcast frames. The converter warns
  when most images are wider than tall.
- Classes can be digits (`0`-`9`, `digit-7`, ...) or whole numbers (`23`, `number-23`); other
  classes such as `jersey` or `player` are ignored, and polygon labels work.
- Roboflow splits images at **random**, so frames of the same clip end up in both train and valid.
  The converter warns when file names suggest that; re-split by game before trusting validation
  accuracy.
- Check the dataset's licence on Roboflow Universe (often CC BY 4.0, which needs attribution if
  you publish the model).

## On a Mac or an NVIDIA PC

From the repo root:

```bash
python colab/make_jersey_labels.py path/to/dataset   # --torso if images are already torso crops
python colab/train_jersey.py path/to/dataset         # same --torso; --arch resnet18 for a lighter model
python colab/train_jersey.py path/to/dataset --arch resnet50 --input-hw 224 176 --batch 64   # the larger model
```

`train_jersey.py` uses an NVIDIA GPU or Apple MPS automatically and saves to
`CV_Models/jersey_Num Models/jersey_model.pt`. On a PC, install PyTorch with CUDA from
[pytorch.org](https://pytorch.org/get-started/locally/) plus `opencv-python pyyaml`; Ultralytics
isn't needed for training.

## Checking the data and results

- `make_jersey_labels.py` prints a summary per split. There should be `train` and `valid` splits,
  roughly 20-40% of crops with no number (or the model never learns to say "none"), and few
  skipped images. Keep each game in one split: frames of the same player in both make validation
  accuracy meaningless.
- Each epoch reports `numbered` (accuracy on crops that show a number, the one to watch), `none`
  (crops without a number correctly read as none) and `balanced` (their mean, which picks the best
  epoch). The end of training lists the most common mistakes.
- Look-alike digits (1↔7, 3↔8) usually mean blur or low resolution: more wide-shot training data
  helps most. Numbers read as "none" can mean the torso crop cuts them off; check the sample crops,
  and if `TORSO_Y` in `jersey_net.py` changes, relabel, retrain and rebuild the engine.
- Model size: `--arch resnet18`, `resnet34` (default) or `resnet50`, and `--input-hw H W` for the
  crop size (default 160 128; e.g. 224 176 for more detail). In the notebook these are `ARCH` and
  `INPUT_HW`. Each checkpoint (and the engine built from it) records its input size, so models of
  different sizes all work without changing any code. Cost per player crop: ResNet-18 at 160x128
  1.5 GFLOPs, ResNet-50 at 224x176 6.5 GFLOPs (the YOLO digit reader: 67.8). On a T4, use
  `BATCH = 64` for ResNet-50 at 224x176 if it runs out of memory. A larger model only helps once
  the training data looks like your footage; compare candidates on the same clips before switching.

## Using the trained model

Put the checkpoint in the repo's `CV_Models/jersey_Num Models/`, either as `jersey_model.pt` or with the notebook's name
(e.g. `jersey_model_resnet18.pt`; the newest one is used), and copy it to the Jetson. There,
inside the container:

```bash
python export_engines.py --models jersey             # builds CV_Models/jersey_Num Models/jersey_model.engine via ONNX
jetson/run_clip.sh videos/<clip>.mp4 --number-reader resnet
```

Compare it with the default PARSeq reader and the YOLO digit reader
(`--number-reader yolo`) on the same clips by the number shown per track
(`tracks.csv`), not by per-crop accuracy.
