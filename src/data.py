# -*- coding: utf-8 -*-

from config import *
import re
from utils import *
from underthesea import word_tokenize, text_normalize
import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib



# text cleaning
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_html_tags(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

def remove_non_alpha(text):
    pattern = re.compile(r'[^a-zA-Z0-9áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđĐ ]')
    return pattern.sub(r' ', text)

# text nomalization
def to_lowercase(text):
    return text.lower()

def standardize_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_text(text):
    # text = remove_urls(text)
    # text = remove_html_tags(text)
    text = remove_non_alpha(text)
    # text = to_lowercase(text)
    text = standardize_spaces(text)
    text = text_normalize(text)
    return text

def top_n_texts(sentences, values_list, n):
    result = []
    for sentence, values in zip(sentences, values_list):
        # Split the sentence into a list of texts
        texts = sentence.split()

        # Create a list of indices
        index_list = list(range(len(texts)))
        
        # Pair the texts and values together
        pairs = list(zip(index_list, values))
        
        # Sort the pairs by values in descending order
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        
        # Get the top n pairs
        top_n_pairs = sorted_pairs[:n]
        
        # Extract the texts from the pairs and retain their original order
        top_n_texts = [texts[idx] for idx, _ in sorted(top_n_pairs, key=lambda x: x[0])]
        
        result.append(' '.join(top_n_texts))
    
    return result

def TF_IDF_truncation(sentences, max_len, vectorizer ):

    # cal idf from your "document".
    X = vectorizer.transform(list(sentences))
    
    tf_idf_values = X.toarray()

    trimmed_sentences = top_n_texts(sentences, tf_idf_values, max_len)

    return trimmed_sentences

def make_dataloader(inputs, labels):

    # Split the data into training and validation sets while maintaining the label distribution
    train_input_ids, val_input_ids, train_attention_mask, val_attention_mask, train_labels, val_labels = train_test_split(
                                                                                                        inputs.input_ids, 
                                                                                                        inputs.attention_mask, 
                                                                                                        labels, 
                                                                                                        test_size= validation_ratio, 
                                                                                                        random_state=42, 
                                                                                                        stratify=labels)

    # Split the data into training and test sets while maintaining the label distribution
    train_input_ids, test_input_ids, train_attention_mask, test_attention_mask, train_labels, test_labels = train_test_split(
                                                                                                        train_input_ids, 
                                                                                                        train_attention_mask, 
                                                                                                        train_labels, 
                                                                                                        test_size= 1 * test_ratio / (1-validation_ratio), 
                                                                                                        random_state=42, 
                                                                                                        stratify=train_labels)


    # Create TensorDatasets for training, validation, and test set
    train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
    val_dataset = TensorDataset(val_input_ids, val_attention_mask, val_labels)
    test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)

    # Create data loaders for training and validation sets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader

def create_data(path_dataset):

    print("=============== Data processing ==============")

    # Read the CSV file
    df = pd.read_excel(path_dataset)
    # Convert all entries in the 'Content' column to strings
    df['Tag'] = df['Tag'].astype(str)
    df['Title'] = df['Title'].astype(str)
    df['Content'] = df['Content'].astype(str)

    # # Filter the dataframe by label 0
    # df_0 = df[df['Label'] == 0]
    # # Filter the dataframe by label 1
    # df_1 = df[df['Label'] == 1]

    # # Take the first 50 rows of each dataframe
    # # df_0 = df_0.head(len(df_1))
    # # df_1 = df_1.head(50)

    # df_0 = df_0.sample(int(len(df_1)*2), random_state=999)
    # # Concatenate the two dataframes
    # df = pd.concat([df_0, df_1], ignore_index=True)


    df['Tag'] = df['Tag'].apply(preprocess_text)
    df['Title'] = df['Title'].apply(preprocess_text)
    df['Content'] = df['Content'].apply(preprocess_text)

    # Tokenize the preprocessed text
    df['Tag'] = df['Tag'].apply(lambda x: word_tokenize(x, format="text"))
    df['Title'] = df['Title'].apply(lambda x: word_tokenize(x, format="text"))
    df['Content'] = df['Content'].apply(lambda x: word_tokenize(x, format="text"))

    # Create an instance of TfidfVectorizer
    vectorizer = TfidfVectorizer()
    # Learn vocabulary and idf from training set.
    vectorizer.fit(df['Content'].to_list())
    # save the vectorizer to disk
    joblib.dump(vectorizer, truncator_path)

    df['Truncated_Content'] = TF_IDF_truncation(df['Content'].to_list(), pretrained_config.max_position_embeddings - 2, vectorizer)

    # Assume df is your DataFrame and it has columns 'Title', 'Tag', 'Content', Label'
    promt_sentences = df.apply(lambda row: prompt_prepare(row['Tag'], row['Title'], row['Truncated_Content']), axis=1)

    tokenizer = AutoTokenizer.from_pretrained(name_model)
    inputs = tokenizer.batch_encode_plus(
                                            promt_sentences, #sentences
                                            max_length=pretrained_config.max_position_embeddings-2, # because [CLS] and [SEP] special tokens
                                            padding='max_length',
                                            truncation=True, 
                                            return_tensors='pt',
                                            add_special_tokens=True
                                        )
    # print(len(inputs.input_ids[0]))
    
    labels = torch.tensor(df['Label'].to_list(), dtype=torch.float)

    train_dataloader, val_dataloader, test_dataloader = make_dataloader(inputs, labels)

    return train_dataloader, val_dataloader, test_dataloader

def info(train_dataloader, val_dataloader, test_dataloader):
    # For the training dataloader
    train_samples = len(train_dataloader.dataset)

    # For the validation dataloader
    val_samples = len(val_dataloader.dataset)

    # For the test dataloader
    test_samples = len(test_dataloader.dataset)

    for batch in train_dataloader:
        print(f"Batch size: {len(batch[0])}")
        print('id_tokens shape: ', batch[0].shape)
        print('attn mask shape: ', batch[1].shape)
        print("label shape: ", batch[2].shape)
        break

    print(f"Number of batches in the training set: {len(train_dataloader)}")
    print(f"Number of samples in the training set: {train_samples}")

    print(f"Number of batches in the validation set: {len(val_dataloader)}")
    print(f"Number of samples in the validation set: {val_samples}")

    print(f"Number of batches in the test set: {len(test_dataloader)}")
    print(f"Number of samples in the test set: {test_samples}")

def infer_input_process(input, truncator, tokenizer):

    # preprocess text: normaliz, etc
    processed_sent = preprocess_text(input.content)

    # word tokenize using underthesea 
    tokenized_text = word_tokenize(processed_sent, format='text')

    # do TF-IDF truncation for longer max leng sentence
    truncated_sent = TF_IDF_truncation(tokenized_text, pretrained_config.max_position_embeddings - 2, truncator)

    input_sentence = prompt_prepare(input.tag, input.title, truncated_sent)

    #  do encode for input text
    input = tokenizer.encode_plus(
                                    input_sentence, #sentences
                                    max_length=pretrained_config.max_position_embeddings-2, # because [CLS] and [SEP] special tokens
                                    padding='max_length',
                                    truncation=True, 
                                    return_tensors='pt'
                                )
    # return input for model
    return input

import transformers
import sys 
if __name__ == "__main__":
    
    path_dataset = "combined_offical_dataset.xlsx"

    train, val = create_data(path_dataset)
    info(train, val)

    name_model = "vinai/phobert-base-v2"
    bert_model = transformers.AutoModel.from_pretrained(name_model)
    bert_tokenizer = transformers.AutoTokenizer.from_pretrained(name_model)
    sys.exit()
    bert_model.to('cuda')

    for batch in train:
        input_ids,  attention_mask, labels = batch 
        
        bert_outputs = bert_model(input_ids.to('cuda'), attention_mask.to('cuda'))
        print(bert_outputs['pooler_output'])
        print(labels.to('cuda'))
        break

