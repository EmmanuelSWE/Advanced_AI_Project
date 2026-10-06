# CRIL replication (reduced)

## What this is
A reduced replication of CRIL (continual learning with imagined replay) on 3 Meta-World robot tasks: reach, button press and drawer open.
It has a CNN policy (image + task in, robot action out), a WGAN-GP first-frame generator (a GAN that makes the first camera image),
and an action-conditioned next-frame predictor (guesses the next image from the current image and an action).

## Setup
1. Create a virtual environment with Python 3.12: `py -3.12 -m venv venv`, then activate it.
2. Install the packages: `pip install -r requirements.txt`
   (metaworld is installed from GitHub, so git and internet are needed).

## Data
The folders `DATASET_generations/` (the PNG frames) and `MODELS_DATASET_CSV/` (the CSV index) must sit at the repo root, next to `src/`.
They are not in git. Download the dataset zip from the Hugging Face dataset
`eashimwe/HE_Ashimwe_222127212_CRIL_REPLICATION_DATASET` and unzip it at the repo root.

## Run
```
cd src
python main.py 1
```
- `1` uses the full data split (demos 1-100). `0` uses the small split (demos 1-7) for quick runs.
- `-l` / `--loadModels`: load the most recent saved models (this option is not working, see Known limits).
- `-s` / `--save`: save the models after each menu item that trains.
- `-t` / `--trainModels`: train only. It runs CRIL training once, saves the models and exits, with no menu.

## Menu
- 1 Train Policy: trains the CNN policy and saves the curves to `policy_Trained.png`.
- 2 Test Policy: tests the policy on the test split and saves `policy_Tested.png`.
- 3 train GAN: trains the WGAN-GP and saves the loss plot `GAN_Trained_Losses.png` and one sample image per epoch in `gen_samples/`.
- 4 train Predictor: trains the next-frame predictor, saves `predictor_Trained.png` and an example picture `predictor_monitor.png`.
- 5 test Predictor: tests the predictor on the test split and saves `predictor_Tested.png`.
- 6 Train CRIL: learns the 3 tasks one after another, replaying earlier tasks with generated data.
- 7 Hyperparameters: view and change training settings.
- 0: quit.

## Hyperparameters
Choose menu item 7. The values are shown in a numbered list. Pick a number, then type a new value (it must be greater than 0).
The new value is used by the next training run. Values go back to the defaults when the program restarts.
The GAN learning rate (1e-4) is not in the list.

## Training workflow
Run the items in this order (the models stay in memory while the program is open):
1. Train Policy (item 1). With `-s` it saves `policy_model.keras`.
2. train GAN (item 3). With `-s` it saves `generator_model.keras` and `critic_model.keras`.
3. train Predictor (item 4). With `-s` it saves `predictor_model.keras`.
4. Train CRIL (item 6). With `-s` it saves `kaggle_policy_model.keras`, `kaggle_generator_model.keras`, `kaggle_critic_model.keras` and `kaggle_predcitor_model.keras`.
Files are written to the folder you run from (`src/`).

## Generation workflow
CRIL training (item 6 or `-t`) imagines trajectories: the generator makes a first frame, the policy picks an action, and the predictor makes the next frame.
For each task, after learning it, it writes (in `src/models/cril_flow.py`):
- `/kaggle/working/cril_results/task_N/` with `policy_model.keras`, `predictor_model.keras`, `generator_model.keras`, `critic_model.keras`.
- In that folder, one sub-folder per learned task with the imagined frames `frame_000.png`, `frame_001.png`, ... (30 steps by default).
- `/kaggle/working/cril_results/testMetrics.json`: test results after each task.
- Temporary replay frames go to `/kaggle/working/cril_replay/` and are deleted after each task.

## Known limits
- CRIL training writes to `/kaggle/working/...` paths set in `src/models/cril_flow.py`, so it is set up for Kaggle. Change those paths to run it elsewhere.
- The `-l` option is not working.
- Frames are 480x480. Training streams the images from disk, and most batch sizes default to 1 (the predictor menu item uses 30) to save memory.
- Collecting new data from Meta-World (`runAndStoreDemonstrations` in `src/utils/meta_wrld_utils.py`) is not in the menu; it must be called by hand.
