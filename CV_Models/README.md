# CV_Models

`run_models.py` needs the five YOLO models below to run the pipeline. They are not in git
(`CV_Models/Models/` is in `.gitignore`); they are stored in the Hugging Face repo
[AlexRHaigh/Hockey-Vision](https://huggingface.co/AlexRHaigh/Hockey-Vision). Download them into
`CV_Models/Models/` before running, from the repo root:

```bash
pip install -U huggingface_hub
hf download AlexRHaigh/Hockey-Vision --include "*.pt" --local-dir CV_Models/Models
```

```
CV_Models/
└── Models/
    ├── new_player_model.pt
    ├── new_nano_puck.pt
    ├── new_dots.pt
    ├── new_rink_model.pt
    └── new_nums.pt
```

| File | `run_models.py` name | Detects |
| --- | --- | --- |
| `new_player_model.pt` | `player` | Players (its `puck` class is ignored; the puck model handles pucks) |
| `new_nano_puck.pt` | `puck` | Pucks (its class `item` is renamed to `puck`) |
| `new_dots.pt` | `dots` | The dots on the ice |
| `new_rink_model.pt` | `rink` | Rink keypoints (used by `homography.py` to map the frame onto the rink) |
| `new_nums.pt` | `number` | Jersey numbers, as single digits `0`-`9` on each player's box |

The file names are set in `MODEL_FILES` in `run_models.py`, and the folder in `MODELS_DIR`.

On the Jetson, `python export_engines.py` builds a TensorRT engine next to each model
(`CV_Models/Models/<name>.engine`), and `run_models.py` uses the engine in place of the `.pt`
from then on. See [jetson/README.md](../jetson/README.md).
