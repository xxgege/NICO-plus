# NICO++
The Official Repository for dataset  NICO++ and Paper "NICO++: Towards Better Benchmarking for Domain Generalization" (https://arxiv.org/abs/2204.08040).

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

