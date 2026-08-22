### SDP Transformer

##### Change to working directory

```
cd ./SPDTransformer
```

##### Set Env Path

Choose a script file corresponding to your operating system to set environment path in the beginning

```powershell
.\script\setup_pythonpath.ps1
```

```bash
source script/setup_pythonpath.sh
```

```cmd
script\setup_pythonpath.cmd
```

#### Prepare dataset

To download BCI Competition IV Dataset 2a (all nine subjects) into `data`:

```powershell
python script/download_bci_competition_iv_2a.py
```

MOABB stores Dataset 2a as ready-to-use `.mat` files, so no manual unzip step
is required. To download only selected subjects, use a list or range such as
`--subjects 1-3,5`.


#### Training

```
python src\training\train.py --config configs/train_grid.yaml --device cuda:0
```

##### Subject-specific PhysioNet four-class benchmark

Train one independent model per subject. Each subject uses an outer five-fold
trial-level split; early stopping uses a validation subset drawn only from the
outer training fold, and the held-out test fold is evaluated once:

```powershell
python src\training\train_subject_specific.py --config configs\train_physionet_subject_specific.yaml --device cuda:0
```

Run a small pilot before launching the full cohort:

```powershell
python src\training\train_subject_specific.py --config configs\train_physionet_subject_specific.yaml --subjects 1-3 --device cuda:0
```

Use `--dry-run` to validate the configuration and show the planned number of
model fits without preprocessing or training. Results include fold-level
metrics, one summary row per subject, an overall mean with between-subject
standard deviation, and a pooled test confusion matrix.

For a fast end-to-end smoke test (not a reportable result), use:

```powershell
python src\training\train_subject_specific.py --config configs\train_physionet_subject_specific.yaml --subjects 1 --outer-splits 2 --epochs 1 --device cpu
```

### baselines
#### MDM
```bash
 python src/baselines/mdm_baseline.py --config configs/mdm_physionet.yaml
```
#### SPDNet
```bash
python src/baselines/spdnet_baseline.py --config configs/spdnet_physionet.yaml --device cuda
```
#### CSP
```bash
python src/baselines/csp_lda_baseline.py   --config configs/csp_lda_physionet.yaml
```

#### BCI Competition IV Dataset 2a

Run each BCI IV-2a baseline separately:

```powershell
python src/baselines/csp_lda_baseline.py --config configs/csp_lda_bci_iv_2a.yaml
python src/baselines/mdm_baseline.py --config configs/mdm_bci_iv_2a.yaml
python src/baselines/spdnet_baseline.py --config configs/spdnet_bci_iv_2a.yaml --device cuda:0
```

Run CSP+LDA, MDM, SPDNet, and SPD Transformer sequentially, then export unified
aggregate and fold-level result tables:

```powershell
python script/run_bci_iv_2a_all_models.py --device cuda:0
```

Use `--dry-run` to generate and inspect the campaign configs without starting
training. Use `--summarize-only <campaign_dir>` to rebuild the CSV and JSON
tables from a completed campaign.

##### Visualize task signals

Plot the complete rest-to-task epoch and every configured frequency band,
grouped by task type. The default motor-cortex channels are C3, Cz, and C4.

```powershell
python script/visualize_task_signals.py --subjects 1-10
```

By default, filter-bank rows show the amplitude envelope so that oscillations
do not cancel when trials are averaged. Use `--band-view waveform` to plot
signed band-pass waveforms, or `--show-trials 5` to overlay individual trials.

##### Four-class result tables

Rerun the main and ablation tables with left hand, right hand, both hands, and
both feet, then export both tables as CSV:

```powershell
python script/run_four_class_experiments.py --device cuda:0
```

Use `--dry-run` to generate and inspect the exact configs without training.
Completed campaigns are stored under `experiments/results/four_class`.


##### Result

Each set of configuration will be saved in

```
experiments/results/spd_transformer_grid/<timestamp>/run_xxx_<hash>/
```

