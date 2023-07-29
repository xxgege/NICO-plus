# NICO++
The Official Repository for dataset NICO++ and CVPR2023 Paper "NICO++: Towards Better Benchmarking for Domain Generalization" (https://arxiv.org/abs/2204.08040).

Acknowledgment: This repository is mostly based on https://github.com/facebookresearch/DomainBed.

![The OOD/DG problem](https://pic.imgdb.cn/item/62592ff2239250f7c5affdd6.jpg?raw=true "Title")
The goal of NICO++ dataset and NICO challenge is to facilitate the OOD (Out-of-Distribution) generalization in visual recognition through promoting the research on the intrinsic learning mechanisms with native invariance and generalization ability. The training data is a mixture of several observed contexts while the test data is composed of unseen contexts. Participants are tasked with developing reliable algorithms across different contexts (domains) to improve the generalization ability of models.

![NICO++ and OOD generalization](https://pic.imgdb.cn/item/625bc201239250f7c5a9893d.png?raw=true "Title")


# Dataset Description
NICO++ dataset is dedicatedly designed for OOD (Out-of-Distribution) image classification. It simulates a real world setting that the testing distribution may induce arbitrary shifting from the training distribution, which violates the traditional I.I.D. hypothesis of most ML methods. The typical research directions that the dataset can well support include but are not limited to Domain Generalization or Domain Adaptation (when testing distribution is known) and General OOD generalization (when testing distribution is unknown).

The basic idea of constructing the dataset is to label images with both main concepts/categories (e.g. dog) and the contexts (e.g. on grass) that visual concepts appear in. By adjusting the proportions of different contexts in training and testing data, one can control the degree of distribution shift flexibly and conduct studies on different kinds of Non-I.I.D. settings.

![Common context in NICO++](https://pic.imgdb.cn/item/62492a8727f86abb2a917846.png?raw=true "Title")
![Unique context in NICO++](https://pic.imgdb.cn/item/62492a8727f86abb2a91785d.png?raw=true "Title")


# Statistics
To boost the heterogeneity and availability of NICO++, the contexts in NICO++ are divided into two types: 1) 10 common contexts that are aligned across all categories, containing nature, season, humanity and light; 2) 10 unique domains specifically for each of the 80 categories, including attributes (e.g. action, color), background, camera shooting angle, and accompanying objects and so on. Totally there are more than 230,000 images with both category and domain label in NICO++.

![NICO++ statistics](https://pic.imgdb.cn/item/625f9bf9239250f7c573ffa5.jpg?raw=true "Title")

# Download
The released data (for NICO challenge) is available at [Dropbox](https://www.dropbox.com/sh/u2bq2xo8sbax4pr/AADbhZJAy0AAbap76cg_XkAfa?dl=0) or here [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/95c45052e2ca41b0ac2e/). You can also free to use NICO++ data for your research for non-economic purpose.

# Sources
The website for NICO++ dataset and NICO challenge is [here](https://nicochallenge.com/).

The NICO challenge on Codalab can be found at:

[Track1: Common Context Generalization](https://codalab.lisn.upsaclay.fr/competitions/4084)

[Track2: Hybrid Context Generalization](https://codalab.lisn.upsaclay.fr/competitions/4083)

# Benchmark
We use [nni](https://nni.readthedocs.io/), an open-source ML toolkit to make the random search in DomainBed protocol more convenient (Note that we only use the random search mode of nni, instead of hyperparameter optimization). 

To run the experiments, you can make a directory named `nni_config_official` in the top directory, and make three directories in it: `autumn_rock`,` dim_grass`, `outdoor_water`, representing the corresponding test domains. For example, to run ERM on the setting where autumn and rock are test domains and the rest four public domains used for training, you can put a yaml file named `ERM.yaml` in `autumn_rock` and write it as:

```yaml
experimentName: NICO_autumn_rock_ERM
searchSpaceFile: ../pretrain.json

trialCommand: python -m domainbed.scripts.train --pretrain --algorithm ERM --dataset NICO --source dim grass outdoor water --target autumn rock --num_classes 60 --seed 59 --checkpoint_freq 1 --steps 10000
trialCodeDirectory: ../..
trialGpuNumber: 1
trialConcurrency: 1
max_trial_number: 10
tuner:
  name: Random
  classArgs:
    seed: 59
trainingService:
  platform: local
  useActiveGpu: True
  gpuIndices: [0]

```

You can change the arguments of`--algorithm`,  `--source` and `--target` to run other experiments. You can change the configs of `trialGpuNumber`, `trialConcurrency`, `gpuIndices` to control the nni properties. 

The search space json file `pretrain.json` should be put in `nni_config_official`. The detailed content is below:

```json
{
    "batch_size": {"_type": "choice", "_value": [32]},
    "lr": {"_type": "loguniform", "_value": [1e-5, 3e-4]},
    "weight_decay": {"_type": "loguniform", "_value": [1e-6, 1e-2]},
    "penalty_anneal_iters": {"_type": "randint", "_value": [1, 5000]},
    "lambda": {"_type": "loguniform", "_value": [1e1, 1e4]},
    "ema": {"_type": "uniform", "_value": [0.9, 0.99]},
    "rsc_f_drop_factor": {"_type": "uniform", "_value": [0, 0.5]},
    "rsc_b_drop_factor": {"_type": "uniform", "_value": [0, 0.5]},
    "irm_lambda":  {"_type": "loguniform", "_value": [1e-5, 1e-1]},
    "irm_penalty_anneal_iters": {"_type": "randint", "_value": [1, 10000]},
    "groupdro_eta":  {"_type": "loguniform", "_value": [1e-3, 1e-1]},
    "mmd_gamma": {"_type": "loguniform", "_value": [1e-1, 1e1]},
    "mixup_alpha": {"_type": "loguniform", "_value": [1e-1, 1e1]},
    "meta_lr": {"_type": "choice", "_value": [0.05, 0.1, 0.5]},
    "sag_w_adv":  {"_type": "loguniform", "_value": [1e-2, 1e1]}
}
```

After writing the yaml files, you can put a script in the top directory to run more experiments automatically:

```shell
port=40500
for alg in "ERM" "SWAD" "RSC" "GroupDRO" "Fishr" "CORAL" "MMD" "SagNet" "IRM" "Mixup" "MixStyle"
do
for domain in "autumn_rock" "dim_grass" "outdoor_water"
do
nnictl create --config nni_config_official_pretrain/$domain/$alg.yaml -p $port
let port++
done
done
```

You can use `nnictl experiment list` to see the  running experiments. You can add another `--all` to see all experiments, including the stopped one. 

You can use port forwarding to see the experiment results by adding in the ssh config and running the command `ssh forward01`. 

```powershell
Host forward01
    HostName xxx
    Port xx
    User xxxxx
    LocalForward 40500 127.0.0.1:40500
```

You may refer to the document of [nnictl](https://nni.readthedocs.io/en/stable/reference/nnictl.html) to know more about it. 

The detailed results of the benchmarks can be found in the [paper](https://arxiv.org/abs/2204.08040).
