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

`python script/download_datasets.py`

Dry Run:  
`python script/download_datasets.py --dry-run`  

PhysionMI dataset Shape

`trial x channels x time samples per trial`



##### Training

```
python src\training\train.py --config configs/train_grid.yaml --device cuda:0
```

##### Result

Each set of configuration will be saved in

```
experiments/results/spd_transformer_grid/<timestamp>/run_xxx_<hash>/
```

Preprocessing pipeline

1. Remove rest state
2. band-pass filter
3. mean value
4. ca

#### Model input

`(batch, epochs(rest+motion), channels, channels)`

### Riemannian Distance

#### Log Map, Foluius

learn a log-like mapping,  implict metric

$$
Z = log(X)
$$

$$
\phi (X) = MLP(Z)

$$

$$
d(X, Y) = ||\phi(X) - \phi(Y)||
$$

#### Metric Network

$$
G_\theta \in SPD 
$$

#### spectural function

[ChatGPT - SPD Transformer Attention设计](https://chatgpt.com/s/t_6a19b10ba6dc8191b77b92378d1b6347)

[ChatGPT - SPD Transformer Attention设计](https://chatgpt.com/s/t_6a19b12be3f481919981d5e5c56201bc)

![58abfc4e-d1b6-41de-9af0-dc46620d26a0](file:///C:/Users/kunsi/Pictures/Typedown/58abfc4e-d1b6-41de-9af0-dc46620d26a0.png)



#### RiemannianBatchNorm
