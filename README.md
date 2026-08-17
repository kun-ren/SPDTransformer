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

To download BCI Competition IV Dataset 2b (all nine subjects) into `data`:

```powershell
python script/download_bci_competition_iv_2b.py
```

MOABB stores Dataset 2b as ready-to-use `.mat` files, so no manual unzip step
is required. To download only selected subjects, use a list or range such as
`--subjects 1-3,5`.


#### Training

```
python src\training\train.py --config configs/train_grid.yaml --device cuda:0
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

