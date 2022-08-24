.PHONY:


#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PYTHONPATH = $(PROJECT_DIR)
LOGGER_NAME = 'eeg_pipeline'
APP_NAME = 'eeg_pipeline'

.EXPORT_ALL_VARIABLES:

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## install requirements
requirements:
	python -m pip install --user -r requirements.txt

## data preprocessing
prep_data:
	python src/data/prep_features.py

## find best model hyperparams
train_model:
	python src/model/train_model.py --multirun 'model.lgbm.boosting_type=choice(gbdt, goss)' \
	 'model.lgbm.num_leaves=range(20, 35)' 'model.lgbm.min_child_samples=range(15, 25)'

## train with best params and predict on test dataset
predict_model:
	python src/model/predict_model.py

## run unit tests
run_test:
	python -m pytest tests --pdb

#################################################################################
# DAGS                                                                          #
#################################################################################

## run full pipeline
all: requirements prep_data train_model predict_model

## run full pipeline except requirements installation
pipe: prep_data train_model predict_model

## run only train model pipeline
model_pipe: train_model predict_model

## run only prep data pipeline
data_pipe: requirements prep_data
