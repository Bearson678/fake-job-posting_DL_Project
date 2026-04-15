To set up the project, create a Python virtual environment and install the required dependencies from ‘requirements.txt’ via the terminal commands:


py -3.10 -m venv .venv310
.\.venv310\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab

Figure 2: Code to create the virtual environment and install dependencies
To train the model from scratch, ensure that “fake_job_postings_ALL.csv” is in the “/data/clean” directory.

To download the clean weights, go to https://drive.google.com/drive/folders/1buxqOhkYUN2XV3OI7jawNvOFpSrOKMNH?usp=sharing. And download both files, inserting it in the root folder of the program. 

Under Model_training.ipynb, run all of the following cells from “Dataset Loading” to “Model Training”.

Below are the markdown cell sections and their functions in notebook order:

Dataset Loading: Makes all necessary imports, sets torch seed and loads in cleaned dataset.
Train-Test-Validation Split: Splits up the dataset and converts value type for tensor conversion.
Non-binary value Standardisation: Standardises non-binary numeric columns
Loading of fine-tuned FastText model: loads the FastText model to extract embedding matrix
Tokenizing and encoding tokens into numerical values: provide numerical representations for the text samples and cache them
Instantiating Datasets: Instantiate Dataset and DataLoader class
Testing Dataloader by sampling a batch: ensures that the custom dataset class is working
Creating the embedding matrix: gets the embeddings from the FastText model and prepares it to be copied into the model
Model Instantiation: Instantiates the model
Model Training: Trains the model
Hyperparameter Threshold Tuning: Hyperparameter tuning of the sigmoid threshold value
Model Branch Evaluation: Compares the performance of both the NLP and numeric branch
Loss and Accuracy Visualisations: Provides visual figures of the results from training
Confusion Matrix Report Visualisation: Provides confusion matrices for evaluation metrics such as precision and recall.
Hyperparameter GRU/Hidden layer dimension tuning: Hyperparameter tuning of the model dimension parameters.

Avoid running the “Hyperparameter GRU/Hidden layer dimension tuning” unless necessary, this will take quite some time to run and our best track record was 5 hours.
