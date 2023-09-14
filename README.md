# melNET-2
This is a tool to train classification model targeting Melanoma-detection in digital images.

## Install requirements
`pip install -r requirements.txt`

## Run training
This pipeline will start training a binary classification-model.

### Check help
`python3 melnet_trainer.py --help`

### Command example
`python3 melnet_trainer.py --config training_config.yaml`

If the process gets `Killed` before MEAN and STD calculation, lower `IMAGE_SET_LIMIT_FOR_MEAN_STD_CALC` value in `./melnet/defaults.py`.

## Generate classification-dataset from MEDNODE
This pipeline will generate classification-dataset from MEDNODE dataset. 

### Download
- Link: https://www.cs.rug.nl/~imaging/databases/melanoma_naevi/ 
- Unzip the downloaded zipped directory.

### Check help
`python3 data_processing/mednode_processing.py --help`

### Command example
`python3 data_processing/mednode_processing.py --mednode_root ../MEDNODE --output_root ../mednode_dataset`

## Generate classification-dataset from HAM10000
This pipeline will generate classification-dataset from HAM10000 dataset. 

### Download
- Link: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
- Unzip all zipped directories from the download.

### Check help
`python3 data_processing/ham_processing.py --help`

### Command example
`python3 data_processing/ham_processing.py --ham_root ../HAM10000 --output_root ../ham_dataset`
