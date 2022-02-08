# Text Classification Comparing Different Models

Here I have solved a text classification problem using the model described in the given  instruction and found the accuracy, precision, recall, and F1 score for the given models. In  addition, I showed the loss curve against epochs.


## Dataset

For the experiment, I used a CSV file named “Dataset.csv”. The dataset link is: https://drive.google.com/file/d/19kmybD6rh63PHeF2AC7ebxPBDcyZE_fU/view
Here I worked with 5791 text data that are binary classified as -1 or 1. In the code you can see full visualization of the of the dataset.


## Pre-Processing Steps

In the given instruction, there are two pre-processing steps are mention.

1. Stop word removal
2. Stemming

For the preprocessing steps I used “nltk” library. From “nltk.corpus” I use “stopwords” function to remove stopward from the dataset and from “nltk.stem.porter” I used “PorterStemmer” function for stemming the dataset.


## Feature Selection

In the given instruction, there are three feature selection method are mention. 
1. BoW (Bag of Word)
2. TF-IDF (Term Frequency-Inverse Document Frequency)
3. One-Hot Embedding

**Bag of Word:** The bag-of-words model is a simplifying representation used in natural language processing and information retrieval. In this model, a text is represented as the bag of its words, disregarding grammar and even word order but keeping multiplicity.
**Used Library:** From sklearn.feature_extraction.text library CountVectorizer

**TF-IDF:** TF-IDF stands for term frequency-inverse document frequency and it is a measure, used in the fields of information retrieval (IR) and machine learning, that can quantify the importance or relevance of string representations (words, phrases, lemmas, etc.) in a document amongst a collection of documents (also known as a corpus).
**Used Library:** From sklearn.feature_extraction.text library TfidfVectorizer

**One-Hot Embedding:** One-Hot Encoding is a general method that can vectorize any categorical features. It is simple and fast to create and update the vectorization, just add a new entry in the vector with a one for each new category.
**Used Library:** From pandas library get_dummies() 


## Model
As per instruction there should be three model. 

1. BoW + Neural Network
2. TF-IDF + Neural Network 
3. One-Hot Embedding + Convolutional Neural Network

### Neural Network

![b](https://user-images.githubusercontent.com/43477718/153038023-11aadca7-c94e-4b8d-8e6a-c4de66bff9bb.png)

### Convolutional Neural Network

![c](https://user-images.githubusercontent.com/43477718/153038271-4c3f5595-90a5-4ac2-b9da-d4bcbd878f4e.png)


## Experiment 
As there are three model, so I conducted three experiment. There are 5791 sample data in the dataset. For all three model, I used 5000 data for training and 791 for testing.

### Experiment 1: 
- Batch size: 50
- Loss function: BCE With Logits Loss
- Optimizer: Adam Optimizer
- Learning Rate: 0.001
- Epoch: 100

### Experiment 2: 
- Batch size: 50
- Loss function: BCE With Logits Loss
- Optimizer: Adam Optimizer
- Learning Rate: 0.001
- Epoch: 100

### Experiment 3:
- Batch size: 50
- Loss function: BCE With Logits Loss
- Optimizer: Adam Optimizer
- Learning Rate: 0.001
- Epoch: 100


## Author

- [@RahatKaderKhan](https://github.com/rahatkader)
