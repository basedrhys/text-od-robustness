# Evaluating the Robustness of Text-Conditioned OD Models to False Captions

## Summary

This repository is the official implementation of [Evaluating the Robustness of Text-Conditioned Object Detection Models to False Captions](#). 

`TODO`:
- Include a graphic explaining the approach/main result
- bibtex entry
- paper summary / abstract
- Include notebook explanation

<p align="center">
  <img width="700" src="./img/incorrect_example.png">
</p>

## Setup

### Python Dependencies

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...
https://github.com/paperswithcode/releasing-research-code

To install requirements:

```setup
pip install -r requirements.txt
```

This project relies on code from the official **MDETR** repo, so you'll need to clone it to your machine:

```setup
git clone https://github.com/ashkamath/mdetr.git
```

### Flickr30k Dataset

Download the **flickr30k** dataset: [link](https://shannon.cs.illinois.edu/DenotationGraph/)

### Script Paths

Finally, update the paths in `eval_flickr.sh` to match your environment:

- `OUTPUT_DIR`: Folder for evaluation results to be saved to
- `IMG_DIR`: **flickr30k** image folder
- `MDETR_GIT_DIR`: Path to **MDETR** repo cloned from above

## Evaluation



The core model evaluation is run via `eval_flickr.sh`:
```
./eval_flickr.sh <batch size> <pretrained model> <gpu type>
```

The currently supported pretrained models are:
- `mdetr_efficientnetB5`
- `mdetr_efficientnetB3`
- `mdetr_resnet101`

For example to evaluate `mdetr_resnet101` with a batch size of 8 while using an RTX8000:

```eval
./eval_flickr.sh 8 mdetr_resnet101 rtx8000
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 
