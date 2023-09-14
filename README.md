# melNET-2
This is a pipeline to train classification model for detecting Melanoma in digital images.

### Install requirements
`pip install -r requirements.txt`

### Check help
`python3 melnet_trainer.py --help`

### Run training
`python3 melnet_trainer.py --config training_config.yaml`

If the process gets `Killed` before MEAN and STD calculation, lower `IMAGE_SET_LIMIT_FOR_MEAN_STD_CALC`
in `melnet/defaults.py`.

## Process HAM10000 dataset
This pipeline will generate classification-dataset from HAM10000 dataset. 

### Check help
`python3 data_processing/ham_processing.py --help`

### Command Example
`python3 data_processing/ham_processing.py --ham_root ../HAM10000 --output_root ../ham_dataset`