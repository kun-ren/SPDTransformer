### SDP Transformer

##### Change to working directory

```
cd ./SPDTransformer
```

##### Set Env Path

Choose a script file corresponding to your operating system to set enironment path in the begin

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

To download and unzip dataset:  

`python script/load_moabb_datasets.py`


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
python src/baselines/spdnet_baseline.py --config configs/spdnet_mi.yaml --device cuda
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


##### Result

Each set of configuration will be saved in

```
experiments/results/spd_transformer_grid/<timestamp>/run_xxx_<hash>/
```

