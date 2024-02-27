import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from data import *
from sklearn.model_selection import GridSearchCV
from optuna import create_study, Trial
import optuna
import os, sys 
from torch.nn import BCEWithLogitsLoss
from time import time
from utils import monitor

class VietnameseFakeNewsDatasetPreprocessor:

    def __init__(self, dataset_path):

        # Read the CSV file
        df  = pd.read_excel(dataset_path)
        # Convert all entries in the 'Content' column to strings
        df['Tag'] = df['Tag'].astype(str)
        df['Title'] = df['Title'].astype(str)
        df['Content'] = df['Content'].astype(str)

        # Filter the dataframe by label 0
        df_0 = df[df['Label'] == 0]
        # Filter the dataframe by label 1
        df_1 = df[df['Label'] == 1]

        ##############################
        df_0 = df_0.sample(min(int(len(df_1)*100), len(df_0)), random_state=999)


        # Take the first 50 rows of each dataframe
        # df_0 = df_0.head(10)
        # df_1 = df_1.head(10)
        
        # Concatenate the two dataframes
        self.dataset = pd.concat([df_0, df_1], ignore_index=True)
        # self.dataset = df

    def preprocess(self):
        """
        Preprocess the dataset for Vietnamese fake news detection.

        This function performs the following steps:
            * Removes punctuation and stop words from the title, tag, and content of the news.
            * Lemmatizes the words in the title, tag, and content of the news.
            * Converts the label column to a binary format (0 for real news, 1 for fake news).

        Returns:
            A Pandas DataFrame containing the preprocessed dataset.
        """
        df = self.dataset
        # Preprocess the 'text' columns
        df['Tag'] = df['Tag'].apply(preprocess_text)
        df['Title'] = df['Title'].apply(preprocess_text)
        df['Content'] = df['Content'].apply(preprocess_text)

        # Tokenize the preprocessed text
        df['Tag'] = df['Tag'].apply(lambda x: word_tokenize(x, format="text"))
        df['Title'] = df['Title'].apply(lambda x: word_tokenize(x, format="text"))
        df['Content'] = df['Content'].apply(lambda x: word_tokenize(x, format="text"))

        # Create an instance of TfidfVectorizer
        trunc_vectorizer = TfidfVectorizer()
        # Learn vocabulary and idf from training set.
        trunc_vectorizer.fit(df['Content'].to_list())
        # save the vectorizer to disk
        # joblib.dump(trunc_vectorizer, truncator_path)

        
        df['Truncated_Content'] = TF_IDF_truncation(df['Content'].to_list(), pretrained_config.max_position_embeddings - 2, trunc_vectorizer)

        # Assume df is your DataFrame and it has columns 'Title', 'Tag', 'Content', Label'
        promt_sentences = df.apply(lambda row: prompt_prepare(row['Tag'], row['Title'], row['Truncated_Content']), axis=1)
        print(promt_sentences)

        # converting the textual data to numerical data
        tokenize_vectorizer = TfidfVectorizer(max_features=1000)
        X = tokenize_vectorizer.fit_transform(promt_sentences.to_list())

        # Convert the label column to a binary format (0 for real news, 1 for fake news).
        y = df['Label'].values

        return X, y
    
def logistic_regression_objective(trial: Trial,  X_train, X_val, y_train, y_val) -> float:    

    params = {
        'tol' : trial.suggest_uniform('tol' , 1e-6 , 1e-3),
        'C' : trial.suggest_loguniform("C", 1e-2, 1),
       'solver' : trial.suggest_categorical('solver' , ['lbfgs','liblinear']),
        "n_jobs" : -1
    }
    model = LogisticRegression(**params, random_state = 9999)
    model.fit(X_train , y_train)
    y_pred = model.predict_proba(X_val)[:,1]
    
    ll = log_loss(y_val , y_pred)
    return ll

def svm_objective(trial: Trial, X_train, X_val, y_train, y_val) -> float:
     

    # Get hyperparameters from trial
    # Define hyperparameters within bounds suggested by Optuna
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
    C = trial.suggest_loguniform("C", 1e-5, 1e5)
    gamma = trial.suggest_loguniform("gamma", 1e-10, 1e10)

    # Train the model
    model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state = 9999)
    model.fit(X_train , y_train)

    # Evaluate the model and return the score
    y_pred = model.predict_proba(X_val)[:,1]
    
    ll = log_loss(y_val , y_pred)

    return ll

def metrices(y_test, y_test_prediction):
    
    '''
    Calculate metrices:
    1. Confusion matrix
    2. Accuracy
    3. Precision
    4. Recall
    5. F1
    6. AUC
    '''
    cm = confusion_matrix(y_test, y_test_prediction)
    accuracy = accuracy_score(y_test, y_test_prediction)
    precision = precision_score(y_test, y_test_prediction)
    recall = recall_score(y_test, y_test_prediction)
    f1 = f1_score(y_test, y_test_prediction)
    auc = roc_auc_score(y_test, y_test_prediction)

    return cm, accuracy, precision, recall, f1, auc
 
def run(model, model_object, X, y, n_trials):
    
    # data split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = validation_ratio, stratify=y, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size =1 * test_ratio / (1-validation_ratio), stratify=y_train, random_state=42)
    

    # initial optuna object
    optuna.logging.set_verbosity(optuna.logging.WARNING) # i do not want to see trail information
    study = optuna.create_study(direction = 'minimize' , study_name = 'lr'
                                , pruner = optuna.pruners.HyperbandPruner(),
                            )
    
    
    # Optimizing
    start_time = time()
    study.optimize(lambda trial: model_object(trial, X_train, X_val, y_train, y_val), n_trials = n_trials)

    num_finished_trials = len(study.trials)
    best_params = study.best_trial.params
    best_value = study.best_value


    # Fit model with the best parameter set
    fited_model = model(**study.best_trial.params)
    fited_model.fit(X_train, y_train)
    training_time = round(time() - start_time, 2)

    # check the fited model with train data
    y_train_prediction = fited_model.predict(X_train)
    train_results = metrices(y_train, y_train_prediction)


    # check the fited model with test data
    start_time = time()
    y_test_prediction = fited_model.predict(X_test)
    test_time = f"{time() - start_time} seconds - with {len(y_test)} samples."
    test_result = metrices(y_test, y_test_prediction)

    return num_finished_trials, best_params, best_value, training_time, test_time, train_results, test_result

def show_case(name_model, results):
    '''
    Args:
        name model: model's name

        results:
        ├── num_finished_trials
        ├── best_params
        ├── best_value
        ├── training_time
        ├── test_time
        ├── train_results
        │   ├── cm
        │   ├── accuracy
        │   ├── precision
        │   ├── recall
        │   ├── f1
        │   └── auc
        └── test_result
            ├── cm
            ├── accuracy
            ├── precision
            ├── recall
            ├── f1
            └── auc

    
    '''
    print(f"\n============= Results of {name_model} =============\n")
    print('Numbers of the finished trials:' , results[0])
    print('The best params:' , results[1])
    

    print("--------- Result on Train ---------")
    print(f"Training time: {results[3]} seconds")
    monitor("Train", None, results[5][0], results[2], results[5][1], results[5][2], results[5][3], results[5][4], results[5][5], None)
    
    print("--------- Result on Test ---------")
    print(f"Infer time: {results[4]}")
    monitor("Test", None, results[6][0], "None", results[6][1], results[6][2], results[6][3], results[6][4], results[6][5], None)


if __name__ == "__main__":

    # Create a preprocessor instance

    print("Loading and processing data ...")
    dataset_preprocessor = VietnameseFakeNewsDatasetPreprocessor(path_data)
    # Preprocess the dataset
    X, y = dataset_preprocessor.preprocess()
    n_trials = 100

    print("Start training and optimizing Logistic Regression ...")
    Logistic_Result = run(LogisticRegression, logistic_regression_objective, X, y, n_trials)

    print("Start training and optimizing Support Vector Machine ...")
    SVM_Result = run(SVC, svm_objective, X, y, n_trials)
    
    print("\n============= Done!!! =============\n")

    show_case("Logistic Regression", Logistic_Result)
    show_case("Support Vector Machine", SVM_Result)


