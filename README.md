# IPG-Rec
The repository for WWW'24 short paper "Proactive Recommendation with Iterative Preference Guidance": https://arxiv.org/abs/2403.07571

To obtain the anaconda environment for the experiment, run

```
conda create --name env --file environment.yaml
```

To reproduce the results, there are 3 steps.

### Step 1: Generate dataset

```
python generate_dataset.py
```

### Step 2: Train a SASRec model on the generated dataset

```
python train_SASRec.py --device cuda:0
```

### Step 3: Using IPG to recommend items on the simulator

```
python test_IPG.py
```

Citation:
```
@inproceedings{bi2024proactive,
  title = {Proactive Recommendation with Iterative Preference Guidance},
  author = {Bi, Shuxian and Wang, Wenjie and Pan, Hang and Feng, Fuli and He, Xiangnan},
  booktitle = {Companion Proceedings of the ACM Web Conference 2024},
  series = {WWW ’24 Companion},
  location = {Singapore, Singapore},
  url = {https://doi.org/10.1145/3589335.3651548},
  doi = {10.1145/3589335.3651548},
  numpages = {4},
  year = {2024}
}
```