# TACaPe: Transformed-based Anti-Cancer Peptide Classification and Generation

TACaPe (Transformed-based Anti-Cancer Peptide Classification and Generation) is a commandline tool
to train transformer-based models for anticancer peptide classification and generation. I was built 
on top of Tensorflow and uses an auto-regressive algorithm for peptide design, which results can
be filtered using an optional classification model.

## Setup

### Installing from PyPI using `pip`

```
$ pip install tacape
```

### Installing from GitHub 

```
$ git clone https://github.com/omixlab/anticancer-peptide
$ cd anticancer-peptide
```

#### Using `pip`

```
$ pip install -r requirements.txt -e .
```

#### Using `conda`

```
$ conda env create
$ conda activate anticancer-peptide
```

## Usage

### `tacape-train-classifier`

Trains a classification model for anticancer peptide.


```
$ tacape-train-classifier -h

/\__  _\ /\  __ \   /\  ___\   /\  __ \   /\  == \ /\  ___\   
\/_/\ \/ \ \  __ \  \ \ \____  \ \  __ \  \ \  _-/ \ \  __\   
   \ \_\  \ \_\ \_\  \ \_____\  \ \_\ \_\  \ \_\    \ \_____\ 
    \/_/   \/_/\/_/   \/_____/   \/_/\/_/   \/_/     \/_____/ 

usage: TACaPe: Model Training [-h] --positive-train POSITIVE_TRAIN --negative-train NEGATIVE_TRAIN --positive-test POSITIVE_TEST
                              --negative-test NEGATIVE_TEST [--format {text,fasta}] --output OUTPUT [--epochs EPOCHS]

optional arguments:
  -h, --help            show this help message and exit
  --positive-train POSITIVE_TRAIN
                        Input file containing positive peptides for training
  --negative-train NEGATIVE_TRAIN
                        Input file containing negative peptides for training
  --positive-test POSITIVE_TEST
                        Input file containing positive peptides for testing
  --negative-test NEGATIVE_TEST
                        Input file containing negative peptides for testing
  --format {text,fasta}
                        [optional] Input file format (default: text)
  --output OUTPUT       Path prefix of the output files
  --epochs EPOCHS       [optional] Number of epochs to be used during training (default: 30)
```

### `tacape-predict`

Runs a classification model for anticancer peptide prediction from a input file.

```
$ tacape-predict -h

/\__  _\ /\  __ \   /\  ___\   /\  __ \   /\  == \ /\  ___\   
\/_/\ \/ \ \  __ \  \ \ \____  \ \  __ \  \ \  _-/ \ \  __\   
   \ \_\  \ \_\ \_\  \ \_____\  \ \_\ \_\  \ \_\    \ \_____\ 
    \/_/   \/_/\/_/   \/_____/   \/_/\/_/   \/_/     \/_____/ 

usage: TACaPe: Predict [-h] --input INPUT [--format {text,fasta}] --classifier-prefix CLASSIFIER_PREFIX --output OUTPUT

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         Input file
  --format {text,fasta}
                        [optional] Input file format (default: text)
  --classifier-prefix CLASSIFIER_PREFIX
                        [optional] Path to the file prefix of the trained classification model
  --output OUTPUT       Path to the output CSV file
```

### `tacape-train-generator`

Trains a auto-regressive generative model for anticancer peptide.

```
$ tacape-train-generator -h

/\__  _\ /\  __ \   /\  ___\   /\  __ \   /\  == \ /\  ___\   
\/_/\ \/ \ \  __ \  \ \ \____  \ \  __ \  \ \  _-/ \ \  __\   
   \ \_\  \ \_\ \_\  \ \_____\  \ \_\ \_\  \ \_\    \ \_____\ 
    \/_/   \/_/\/_/   \/_____/   \/_/\/_/   \/_/     \/_____/ 

usage: TACaPe: Generative Model Training [-h] --positive-train POSITIVE_TRAIN --positive-test POSITIVE_TEST [--format {text,fasta}] --output
                              OUTPUT [--epochs EPOCHS]

optional arguments:
  -h, --help            show this help message and exit
  --positive-train POSITIVE_TRAIN
                        Input file containing positive peptides for training
  --positive-test POSITIVE_TEST
                        Input file containing positive peptides for testing
  --format {text,fasta}
                        [optional] Input file format (default: text)
  --output OUTPUT       Path prefix of the output files containing the trained model
  --epochs EPOCHS       [optional] Number of epochs to be used during training (default: 30)
```

### `tacape-generate`

Generates a set of peptides with potential anticancer activity from a trained generative model. If a classification
model é provided, it will be used to filter the generated sequences and compute a probability of activity.

```
$ tacape-generate -h

/\__  _\ /\  __ \   /\  ___\   /\  __ \   /\  == \ /\  ___\   
\/_/\ \/ \ \  __ \  \ \ \____  \ \  __ \  \ \  _-/ \ \  __\   
   \ \_\  \ \_\ \_\  \ \_____\  \ \_\ \_\  \ \_\    \ \_____\ 
    \/_/   \/_/\/_/   \/_____/   \/_/\/_/   \/_/     \/_____/ 

usage: TACaPe: Generate [-h] --generator-prefix GENERATOR_PREFIX [--classifier-prefix CLASSIFIER_PREFIX]
                        [--number-of-sequences NUMBER_OF_SEQUENCES] [--temperature TEMPERATURE] [--threshold THRESHOLD] --output
                        OUTPUT

optional arguments:
  -h, --help            show this help message and exit
  --generator-prefix GENERATOR_PREFIX
                        Path to the file prefix of the trained generative model
  --classifier-prefix CLASSIFIER_PREFIX
                        [optional] Path to the file prefix of the trained classification model
  --number-of-sequences NUMBER_OF_SEQUENCES
                        [optional] Number of sequences to be generated (default: 1000)
  --temperature TEMPERATURE
                        [optional] Temperature used for logit scaling when sampling aminoacids during auto-regressive generation
                        (default: 1.0)
  --threshold THRESHOLD
                        [optional] Classification probability threshold (default: 0.5)
  --output OUTPUT       Path to the output CSV file
```

## Example: generating sequences from the AntiCP2 dataset

### Creating a peptide classifier for 100 epochs

```
$ tacape-train-classifier \
    --positive-train data/raw/anti_cp/anticp2_main_internal_positive.txt \
    --negative-train data/raw/anti_cp/anticp2_main_internal_negative.txt \
    --positive-test data/raw/anti_cp/anticp2_main_validation_positive.txt \
    --negative-test data/raw/anti_cp/anticp2_main_validation_negative.txt \
    --output data/models/classifier \
    --epochs 100
```

### Run the predictive model on the validation dataset

```
$ tacape-predict \
    --input data/raw/anti_cp/anticp2_main_validation_positive.txt \
    --format text \
    --classifier-prefix data/models/internal \
    --output data/models/internal_results.csv
```

### Creating a peptide generator for 100 epochs

```
$ tacape-train-generator \
    --positive-train data/raw/anti_cp/anticp2_main_internal_positive.txt \
    --positive-test data/raw/anti_cp/anticp2_main_validation_positive.txt \
    --output data/models/generator \
    --epochs 100
```

### Run the generative model to generate 100 sequences

```
$ tacape-generate \
    --generator-prefix data/models/generator \
    --classifier-prefix data/models/classifier \
    --number-of-sequence 100 \
    --output data/models/generated.csv
```