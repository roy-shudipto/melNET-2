# melNET-2
This is a pipeline to train classification model for detecting Melanoma in digital images.

### Install requirements
`pip install -r requirements.txt`

### Check help
`python3 melnet_trainer.py --help`

### Run training
`python3 melnet_trainer.py --config training_config.yaml`

## Process HAM10000 dataset
This pipeline will generate classification-dataset from HAM10000 dataset. 

### Check help
`python3 data_processing/ham_processing.py --help`

### Command Example
`python3 data_processing/ham_processing.py --ham_root ../HAM10000 --output_root ../ham_dataset`