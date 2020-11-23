# Humorous Headline Generation via Style Transfer (a.k.a. Humor Translation)
Code and Datasets from the paper, ["Can Humor Prediction Datasets be used for Humor Generation? Humorous Headline Generation via Style Transfer"](https://www.aclweb.org/anthology/2020.figlang-1.25/) by Orion Weller, Nancy Fulda, and Kevin Seppi.

## Related Work
For related projects, see our work on [Humor Detection (separating the humorous jokes from the non-humorous)](https://github.com/orionw/RedditHumorDetection) or our [collection of 500k+ jokes](https://github.com/orionw/rJokesData/blob/master/README.md).

** **We do not endorse these jokes. Please view at your own risk** **

## Overview
Data files are located in `data/*.csv` and are from the [HumicroEdit Dataset](https://www.cs.rochester.edu/u/nhossain/humicroedit.html). The script to reproduce the dataset files is `create_sentences.py`. The neural translation model is in `transformer.py` while the random model comes from the `random_translate.py` file.

# How to Use
## Set Up
0. Run `pip3 install -r requirements.txt` and `python -m spacy download en` to install the correct packages

## Train the Model
0. Run `python3 translate.py` to start producing translations.  If you want to enter in your own sentences interactively after each epoch, use `python3 translate.py --interactive`

## Evaluate Trained Models:
0. Run `python3 translate.py -all -e -s` to generate a file with outputs
1. Run `python3 clean_and_ready_eval.py` to add in the Random model and clean up puncuation, etc.

# Pre-trained Models
I've included four pre-trained models at various epochs (30, 60, 90, 120). If you'd like to just play with a model, feel free to use one of them (found in the `models/{epoch_num}-{timestamp}-model.pt`).

# Citation
If you found this repository helpful, please cite the following paper:

```
@inproceedings{weller2020can,
  title={Can Humor Prediction Datasets be used for Humor Generation? Humorous Headline Generation via Style Transfer},
  author={Weller, Orion and Fulda, Nancy and Seppi, Kevin},
  booktitle={Proceedings of the Second Workshop on Figurative Language Processing},
  pages={186--191},
  year={2020}
}
```