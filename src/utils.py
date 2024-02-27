import torch
import joblib
from config import *
from transformers import AutoTokenizer
import model as model_class
from termcolor import cprint
from tabulate import tabulate
import numpy as np

# Define the save_checkpoint and load_checkpoint functions
def save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path):
    # Save the state of the model, optimizer, and current epoch to the specified checkpoint file
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss
    }, checkpoint_path)
    print('\tSaved checkpoint!', end='')

def load_checkpoint(model, optimizer, checkpoint_path):
    # Load the state of the model, optimizer, and current epoch from the specified checkpoint file
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']
    # Return the current epoch
    return model, optimizer, epoch, val_loss

def get_preds(logits):
    
    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    # Get the predicted labels
    probs, predicted_labels = torch.max(probabilities, dim=-1)
    
    return probs, predicted_labels

# Prepare prompt data
def prompt_prepare(tag, title, text):
    prompt = f"{tag} - {title}: {text}"
    return prompt

def load_model():
    
    # create model
    model = model_class.FakeNewsClassifier(pretrained_config.hidden_size, dense_size, num_labels)

    # Load the checkpoint
    checkpoint = torch.load(infer_ckpt)

    # Load model and optimizer states
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    model.to(device)
    
    return model

def load_truncator():

    # later, load the vectorizer from disk
    vectorizer = joblib.load(truncator_path)

    return vectorizer 


def load_tokenizer():

    tokenizer = AutoTokenizer.from_pretrained(name_model)

    return tokenizer

def monitor(name, epoch, cm, avg_loss, accuracy, precision, recall, f1, auc, writer):
    '''
    Provide informations after each epoch
    '''

    # Print confusion matrix

    np.set_printoptions(precision=2, suppress=True)  # Adjust precision as needed
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            color = 'green' if i == j else 'red'  # Adjust colors as desired
            cprint(cm[i, j], color=color, end=' ')
        print()

    print("-" * 50)
    print("Confusion matrix:")
    print(tabulate(cm, headers='firstrow', tablefmt='fancy_grid'))

    # Print other metrices

    metrices = {
                    f"{name} Loss": avg_loss,
                    f"{name} Accuracy": accuracy,
                    f"{name} Precision": precision,
                    f"{name} Recall": recall,
                    f"{name} F1": f1,
                    f"{name} AUC": auc
                }

    print("-" * 50)
    for metric, value in metrices.items():
        try:
            print(f"{metric:20s}: {value:.4f}")
        except:
            print(f"{metric:20s}: Not provided!")
    print("-" * 50)

    if 'Test' not in name and writer is not None:
        writer.add_scalar(f"Loss/{name}_epoch", avg_loss, epoch)
        writer.add_scalar(f"Accuracy/{name}_epoch", accuracy, epoch)
        writer.add_scalar(f"Precision/{name}_epoch", precision, epoch)
        writer.add_scalar(f"Recall/{name}_epoch", recall, epoch)
        writer.add_scalar(f"F1/{name}_epoch", f1, epoch)
        writer.add_scalar(f"AUC/{name}_epoch", auc, epoch)

