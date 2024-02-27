import model as model_class
import data
from config import *
import utils

from time import time
import torch
import sys, os
import datetime
import argparse
import numpy as np
from  tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from transformers import AdamW
from torch.nn import BCEWithLogitsLoss

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix




def train_func(model, optimizer, loss_fn, train_dataloader):

    
    # Training
    total_loss = 0
    
    correct_predictions = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(train_dataloader, ncols=100, desc="batch: "):
        
        model.train()
        torch.cuda.empty_cache()
        optimizer.zero_grad()

        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        loss = loss_fn(outputs[:, 1], labels)
        
        total_loss += loss.item()
        loss.backward()

        # do some tricks preventing exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 

        optimizer.step()

        
        model.eval()
        with torch.no_grad():
            probs, preds = utils.get_preds(outputs)
            correct_predictions += torch.sum(preds == labels)

            # Store all predictions and labels for calculating metrics later
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
    avg_train_loss = total_loss / len(train_dataloader)
    train_accuracy = correct_predictions.double() / len(train_dataloader.dataset)

    # Calculate confusion matrix
    train_cm = confusion_matrix(all_labels, all_preds)

    # Calculate precision, recall, F1-score, and AUC
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)

    return train_cm, avg_train_loss, train_accuracy, precision, recall, f1, auc

def val_func(model, loss_fn, val_dataloader):
    # Validation
    model.eval()
    val_loss = 0
    correct_predictions = 0
    all_preds = []
    all_labels = []

    start_time = time()
    
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            loss = loss_fn(outputs[:, 1], labels)
            val_loss += loss.item()
            
            probs, preds = utils.get_preds(outputs)
            correct_predictions += torch.sum(preds == labels)

            # Store all predictions and labels for calculating metrics later
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print(f"Infer time: {round(time() - start_time, 2)} seconds - with {len(all_labels)} samples.")


    avg_val_loss = val_loss / len(val_dataloader)
    val_accuracy = correct_predictions.double() / len(val_dataloader.dataset)

    # Calculate confusion matrix
    val_cm = confusion_matrix(all_labels, all_preds)


    # Calculate precision, recall, F1-score, and AUC
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)

    return val_cm, avg_val_loss, val_accuracy, precision, recall, f1, auc


def train_and_evaluate():
    
    # load dataset
    train_dataloader, val_dataloader, test_dataloader = data.create_data(path_data)
    # check dataset info
    data.info(train_dataloader, val_dataloader, test_dataloader)

    print("\n=============== Preparing dependence resources ===============")
    
    # create model
    model = model_class.FakeNewsClassifier(pretrained_config.hidden_size, dense_size, num_labels)

    # setup train on multi GPU
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model, dim=0)
    model.to(device)

    # Define your optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    loss_fn = BCEWithLogitsLoss(pos_weight=class_weights)
    # loss_fn = nn.NLLLoss()

    #load if checkpoint
    trained_epoch = 0
    best_epoch = 0
    pre_loss = 10**9
    if os.path.isfile(ckpt_path):
        print(f"Load model from check point: {ckpt_path}")
        _, _, trained_epoch, pre_loss = utils.load_checkpoint(model, optimizer, ckpt_path)
        best_epoch = trained_epoch

    # Create a SummaryWriter instance for TensorBoard logging
    train_writer = SummaryWriter(log_dir=train_dir)
    val_writer = SummaryWriter(log_dir=val_dir)
    
    # Training and evaluating model
    print("\n=============== Start Training Phase ===============")
    start_time = time()

    for epoch in range(trained_epoch + 1, trained_epoch + num_epochs + 1):
        print(f'EPOCH {epoch}/{trained_epoch + num_epochs}:')

        # Do train
        train_cm, avg_train_loss, train_accuracy, train_precision, train_recall, train_f1, train_auc = train_func(model, optimizer, loss_fn, train_dataloader)
        utils.monitor("Train", epoch, train_cm, avg_train_loss, train_accuracy, train_precision, train_recall, train_f1, train_auc, train_writer)


        # Do validate
        val_cm, avg_val_loss, val_accuracy, val_precision, val_recall, val_f1, val_auc = val_func(model, loss_fn, val_dataloader)
        utils.monitor("Validation", epoch, val_cm, avg_val_loss, val_accuracy, val_precision, val_recall, val_f1, val_auc, val_writer)
        
        #check best epoch
        if avg_val_loss < pre_loss:
            pre_loss = avg_val_loss
            best_epoch = epoch
            checkpoint_path = f"{ckpt_dir}/FND_lr_{str(learning_rate)}_dense_{dense_size}_batchsize_{batch_size}_epoch_{epoch}_vloss_{avg_val_loss:.4f}_acc_{val_accuracy:.4f}_precision_{val_precision:.4f}_recall_{val_recall:.4f}_F1_{val_f1:.4f}_AUC_{val_auc:.4f}.pt"
            utils.save_checkpoint(model, optimizer, epoch, avg_val_loss, checkpoint_path)
        print('\n')

    train_writer.close()
    val_writer.close()

    end_time = time() - start_time
    print(f'Training take: {round(end_time, 3)} seconds ~ {round(end_time/60, 3)} minutes ~ {round(end_time/3600, 3)} hours')
    print(f'Best Epoch: {best_epoch} with val_loss = {round(pre_loss, 8)}')

    print("\n=============== Start Testing Phase ===============")
    #start_time = time()
    test_cm, avg_test_loss, test_accuracy, test_precision, test_recall, test_f1, test_auc = val_func(model, loss_fn, test_dataloader)
    #print(f"Infer time: {round(time() - start_time, 2)} seconds - with {len(test_dataloader.dataset)} samples.")
    utils.monitor("Test", None, test_cm, avg_test_loss, test_accuracy, test_precision, test_recall, test_f1, test_auc, None)
  

if __name__=='__main__':

    train_and_evaluate()





