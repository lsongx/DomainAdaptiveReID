# Unsupervised Domain Adaptive Re-Identification

Implementation of the paper [Unsupervised Domain Adaptive Re-Identification: Theory and Practice](https://arxiv.org/abs/1807.11334). 

The selftraining scheme proposed in the paper is simple yet effective.

![Illustration of the selftraining scheme.](./data/algorithm_illustration.png)

## Setup

1. Datasets (source dataset and target dataset).
2. Pre-trained (on source dataset) model.

## Requirements

- PyTorch

## Running the experiments

To replicate the results in the paper, you can download pre-trained models on Market1501 from [GoogleDrive](https://drive.google.com/open?id=1xNqduSroUMDbM_E5VeeR1WuykMh8Oxlb) and on DukeMTMC from [GoogleDrive](https://drive.google.com/file/d/1CFuf_vF9OphbuCyMefa3W8GA8tgcvSkI/view?usp=sharing). Our models are trained with __PyTorch 0.3__.

<!-- > Code is temporarily removed in the latest commit due to a bug of memory leak. Please check the commit history. If you have any questions, please contact liangchen.song AT horizon.ai. -->

```
python selftraining.py \
    --src_dataset <name_of_source_dataset>\
    --tgt_dataset <name_of_target_dataset>\
    --resume <dir_of_source_trained_model>\
    --data_dir <dir_of_source_target_data>\
    --logs_dir <dir_to_save_model_after_adaptation>
```

`dw_example.ipynb` is the file for replicating Figure 6 in the paper.

### Results

#### DukeMTMC ---> Market1501

| | Rank-1 | Rank-5 | Rank-10| mAP|
| --- | :---: | :---: | :---: | :---: |
|On source (DukeMTMC)| 80.8 | 91.2 | 94.2 | 65.4|
|On target (Market1501)| 46.8|64.6|71.5|19.1|
|After adaptation| 75.8|89.5|93.2|53.7|

#### Market1501 ---> DukeMTMC

| | Rank-1 | Rank-5 | Rank-10| mAP|
| --- | :---: | :---: | :---: | :---: |
|On source (Market1501)| 91.6 | 97.1 | 98.5 | 78.2|
|On target (DukeMTMC)| 27.3|41.2|47.1|11.9|
|After adaptation| 68.4|80.1|83.5|49.0|

<!-- #### Market1501 --- > CUHK03 -->

<!-- | | Rank-1 | Rank-5 | Rank-10| mAP| -->
<!-- | --- | :---: | :---: | :---: | :---: | -->
<!-- |On source (Market1501)| 91.6 | 97.1 | 98.5 | 78.2| -->
<!-- |On target (CUHK03)| 11.5|23.5|34.5|9.0| -->
<!-- |After adaptation| 38.0|59.5|69.0|28.9| -->

## Acknowledgement

Our code is based on [open-reid](https://github.com/Cysu/open-reid).