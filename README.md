# Project Setup and Training Guide

## Environment Setup

Create a Python virtual environment and install the required dependencies from `requirements.txt`.

```bash
py -3.10 -m venv .venv310
.\.venv310\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

## Training From Scratch

To train the model from scratch, ensure that:

- `fake_job_postings_ALL.csv` is placed inside the `data/clean/` directory

Then open `Model_training.ipynb` and run all cells from **Dataset Loading** through **Model Training**.

## Downloading Pretrained Weights

To use the pre-trained model weights instead of training from scratch:

1. Go to the Google Drive folder below:  
   [Download pretrained weights](https://drive.google.com/drive/folders/1buxqOhkYUN2XV3OI7jawNvOFpSrOKMNH?usp=sharing)

2. Download both files

3. Place both files in the **root directory** of the project

## Notebook Workflow

Below are the main markdown sections in `Model_training.ipynb` and what each one does.

### 1. Dataset Loading
- Imports all required libraries
- Sets the Torch seed
- Loads the cleaned dataset

### 2. Train-Test-Validation Split
- Splits the dataset into training, validation, and test sets
- Converts values into suitable types for tensor conversion

### 3. Non-binary Value Standardisation
- Standardises non-binary numeric columns

### 4. Loading of Fine-tuned FastText Model
- Loads the FastText model
- Extracts the embedding matrix

### 5. Tokenizing and Encoding Tokens into Numerical Values
- Converts text samples into numerical representations
- Caches the processed outputs

### 6. Instantiating Datasets
- Instantiates the custom `Dataset` and `DataLoader` classes

### 7. Testing Dataloader by Sampling a Batch
- Verifies that the custom dataset class is working correctly

### 8. Creating the Embedding Matrix
- Retrieves embeddings from the FastText model
- Prepares them to be copied into the model

### 9. Model Instantiation
- Instantiates the model architecture

### 10. Model Training
- Trains the model

### 11. Hyperparameter Threshold Tuning
- Tunes the sigmoid threshold value

### 12. Model Branch Evaluation
- Compares the performance of the NLP branch and numeric branch

### 13. Loss and Accuracy Visualisations
- Generates visualisations of training results

### 14. Confusion Matrix Report Visualisation
- Displays confusion matrices and evaluation metrics such as precision and recall

### 15. Hyperparameter GRU / Hidden Layer Dimension Tuning
- Tunes model dimension-related hyperparameters

## Important Note

Avoid running **Hyperparameter GRU / Hidden Layer Dimension Tuning** unless necessary.  
This step is computationally expensive and, in our best recorded run, took around **5 hours**.
