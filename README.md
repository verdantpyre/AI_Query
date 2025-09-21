# Multilingual Query-Category Relevance System

A machine learning system designed to map user search queries to the most relevant product categories in a large scale, multilingual e-commerce system. This project addresses the "Query–Category Relevance" task to improve search engine performance and user experience.

## Project Overview

Global e-commerce platforms (e.g., Amazon, Walmart) serve millions of users across diverse languages and regions. A core challenge is understanding user intent from a search query to return product results. This system tackles that by:

*   **Analyzing** a user's search query (e.g., “red running shoes”).
*   **Mapping** it to the most relevant hierarchical category path (e.g., `Sportswear > Footwear > Running Shoes`).
*   **Operating** across multiple languages.

## Project Structure

├── main.py              # Production classifier for predictions  
├── trainer.py           # Model training   and optimization  
├── evaluation.py        # Model performance evaluation  
├── preprocessing.py     # Text cleaning and normalization  
├── gui.py              # Graphical user interface  
├── models/              # Directory for trained models  
│   ├── model.bin       # Pre-trained model (example)  
│   └── rfcOPT.bin      # Optimized model (example)  
├── data/               # Data directory (not included in repo)  
│   ├── train.csv       # Training data  
│   └── test.csv        # Test data  
└── README.md           # This file  

## Model Architecture
	Text Processing: Custom sanitization and normalization  
    Feature Extraction: TF-IDF with n-grams (1-2)  
    Classifier: Random Forest with 100 estimators  
    Optimization: Optional hyperparameter tuning available  

## How to install

To use a pre-trained model, install the corresponding model.bin file  
To train a model on your own data set, install trainer.py and evaluation.py excute them as such:  

python trainer.py [training_data_path] [model_output_path] [language] [hyperparameter_tuning]  

Default values:  
    training_data_path: data/train.csv  
    model_output_path: models/model.bin  
    language: all (process all languages)  
    hyperparameter_tuning: False  

To evaluate the accuracy of a model:  

python evaluation.py [test_data_path] [model_path] [language]  

Default values:  
    test_data_path: data/test.csv  
    model_path: models/model.bin  
    language: all  

When satisfied, you can create an exe file for the model using Pyinstaller and gui.py  
Have gui.py and models directory in the same folder  

cd path/to/folder  
python -m venv pyinstaller-env  
pyinstaller-env\Scripts\activate  
pip install pyinstaller  
pip install scikit-learn  
pip install pandas //also installs numpy  
pyinstaller --noconsole --onefile -- "models/model/used;models" gui.py  
./dist/gui.exe  

main.py provides an in-terminal way for individual predictions  
python main.py  

### Prerequisites
- Python 3.8+  
- Required packages:  
  pip install pandas==2.3.2 numpy==2.3.3 scikit-learn==1.7.2  

## Technologies Used
Python  
numpy, pandas, scikit-learn, pyinstaller  

## Contributors
aditya [dot] pranjal [dot] cse25 [at] itbhu [dot] ac [dot] in  
saumil [dot] yadav [dot] cse25 [at] itbhu [dot] ac [dot] in  
aditya [dot] aggarwal [dot] ee25 [at] itbhu [dot] ac [dot] in  
arnav [dot] pundir [dot] cse25 [at] itbhu [dot] ac [dot] in  
