# MLM trainer
Scripts for training a masked language model from scratch. Either BERT or RoBERTa can be chosen.

## Instructions

1. Download and preprocess some data with `source get_and_preprocess_data.sh`  _this will download english, german, estonian and finnish parallel datasets into .`/data/`_
2. Train a model with the `hf_train_simpleBERTfromscratch.sh` script.

__Remember to modify the paths needed in these two scripts__
### Requirements

For the trainer requirements, please see the `trainer_dependencie.md` file.

For the data preprocessing, you'll need:
```
pip install opustools sentencepiece
```
