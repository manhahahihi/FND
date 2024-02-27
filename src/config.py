from transformers import AutoConfig
import torch
import os


path_data = "combined_offical_dataset.xlsx"

# Load the configuration of the model
name_model = "vinai/phobert-base-v2"

pretrained_config = AutoConfig.from_pretrained(name_model)
print(pretrained_config)



validation_ratio = 0.2
test_ratio = 0.1

# Specify the number of labels for sequence classification
num_labels = 2

num_epochs = 7

batch_size = 16

dense_size = 512

learning_rate = 5e-5

class_weights = None

# best: batch size 16, dense 512, lr 5e-5, add_special_tokens=True, output from pooler_layer


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

ckpt_path = ""
train_dir = "train_log"
val_dir = "val_log"
ckpt_dir = "ckpt"

if not os.path.exists(train_dir):
    os.makedirs(train_dir)

if not os.path.exists(val_dir):
    os.makedirs(val_dir)

if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)


# inference config

infer_ckpt = "D:\KLTN_FNDd\KLTN-FND--main\ckpt\FND_lr_5e-05_dense_512_batchsize_16_epoch_6_vloss_0.0514_acc_0.9775_precision_0.9879_recall_0.9702_F1_0.9790_AUC_0.9782.pt"

truncator_path = 'resources/tfidf_vectorizer.joblib'

root_part = ""

api_port = 8000
api_url = f"http://localhost:{api_port}/predict"

web_port = 5555
