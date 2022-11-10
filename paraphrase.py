import os, sys
# sys.path.append('../')
# os.chdir('../')

import torch
import shutil
import random
import datasets
import numpy as np
import pandas as pd
from torch import optim
from torch.utils.data import Dataset, DataLoader
from transformers import MBartForConditionalGeneration

from modules.tokenization_indonlg import IndoNLGTokenizer
from utils.train_eval import train, evaluate
from utils.metrics import generation_metrics_fn
from utils.forward_fn import forward_generation
from utils.data_utils import MachineTranslationDataset, GenerationDataLoader

import nltk
nltk.download('punkt')

###
# common functions
###
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
def count_param(module, trainable=False):
    if trainable:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in module.parameters())
    
# Set random seed
set_seed(42)

bart_model = MBartForConditionalGeneration.from_pretrained('indobenchmark/indobart')
tokenizer = IndoNLGTokenizer.from_pretrained('indobenchmark/indobart')
model = bart_model

# configs and args

lr = 1e-4
gamma = 0.9
lower = True
step_size = 1
beam_size = 5
max_norm = 10
early_stop = 5
# early_stop = 2 # filtered_paracotta
# early_stop = 1 # full_paracotta

max_seq_len = 128
grad_accumulate = 1
no_special_token = False
swap_source_target = True
model_type = 'indo-bart'
valid_criterion = 'BERTSCORE'

separator_id = 4
speaker_1_id = 5
speaker_2_id = 6

train_batch_size = 8
valid_batch_size = 8
test_batch_size = 8

source_lang = "[indonesian]"
target_lang = "[indonesian]"

optimizer = optim.Adam(model.parameters(), lr=lr)
src_lid = tokenizer.special_tokens_to_ids[source_lang]
tgt_lid = tokenizer.special_tokens_to_ids[target_lang]

model.config.decoder_start_token_id = tgt_lid

# Make sure cuda is deterministic
torch.backends.cudnn.deterministic = True

# create directory
# model_dir = './save/filtered_liputan6-indolem'
# model_dir = './save/filtered_paracotta'
# model_dir = './save/full_paracotta'
# model_dir = './save/full_liputan6-merge'
model_dir = './save/full_liputan6-indolem'
if not os.path.exists(model_dir):
    os.makedirs(model_dir, exist_ok=True)

device = "cuda0"
# set a specific cuda device
if "cuda" in device:
    torch.cuda.set_device(int(device[4:]))
    device = "cuda"
    model = model.cuda()

PATH = "/workspace/bertshare"
MAIN_PATH = PATH+"/paraphrase"
# dataset_conf: dict = {
#     "path": "csv",
#     "data_dir": MAIN_PATH+"/data",
#     "data_files": MAIN_PATH+"/data/filtered_liputan6-indolem.csv"
# }
# col1 = "summary"
# col2 = "generated_summary"

# dataset_conf: dict = {
#     "path": "csv",
#     "data_dir": MAIN_PATH+"/data",
#     "data_files": MAIN_PATH+"/data/filtered_paracotta.csv"
# }
# col1 = "references"
# col2 = "paraphrase"

# dataset_conf: dict = {
#     "path": "csv",
#     "data_dir": MAIN_PATH+"/data",
#     "data_files": MAIN_PATH+"/data/full_paracotta.csv"
# }
# col1 = "references"
# col2 = "paraphrase"

# dataset_conf: dict = {
#     "path": "csv",
#     "data_dir": MAIN_PATH+"/data",
#     "data_files": MAIN_PATH+"/data/full_liputan6-merge.csv"
# }
# col1 = "indolem_summary"
# col2 = "indonlu_summary"

dataset_conf: dict = {
    "path": "csv",
    "data_dir": MAIN_PATH+"/data",
    "data_files": MAIN_PATH+"/data/full_liputan6-indolem.csv"
}
col1 = "summary"
col2 = "generated_summary"

class ParaphraseDataset(Dataset):
    
    def load_dataset(self, dataset_conf): 
        data = datasets.load_dataset(split="train", **dataset_conf)
        # data = data.rename_column("Unnamed: 0", "id")
        data = data.rename_column("bert_score", "id")
        data = data.rename_column(col1, "text")
        data = data.rename_column(col2, "label")
        return data

    def __init__(self, dataset_conf, tokenizer, swap_source_target, is_valid=False, *args, **kwargs):
        self.data = self.load_dataset(dataset_conf)
        if not is_valid:
            self.data = self.data.select(range(0, self.data.num_rows-100))
            # self.data = self.data.select(range(16))
        else:
            self.data = self.data.select(range(self.data.num_rows-100, self.data.num_rows))
            # self.data = self.data.select(range(16))
        self.tokenizer = tokenizer
        self.swap_source_target = swap_source_target
    
    def __getitem__(self, index):
        data = self.data[index]
        id, text, label = data['id'], data['text'], data['label']
        input_subwords = self.tokenizer.encode(text.lower(), add_special_tokens=False)
        label_subwords = self.tokenizer.encode(label.lower(), add_special_tokens=False)
        if self.swap_source_target:
            return data['id'], label_subwords, input_subwords
        else:
            return data['id'], input_subwords, label_subwords
    
    def __len__(self):
        return len(self.data)

train_dataset = ParaphraseDataset(dataset_conf, tokenizer, is_valid=False, lowercase=lower, no_special_token=no_special_token, 
                                            speaker_1_id=speaker_1_id, speaker_2_id=speaker_2_id, separator_id=separator_id,
                                            max_token_length=max_seq_len, swap_source_target=swap_source_target)
valid_dataset = ParaphraseDataset(dataset_conf, tokenizer, is_valid=True, lowercase=lower, no_special_token=no_special_token, 
                                            speaker_1_id=speaker_1_id, speaker_2_id=speaker_2_id, separator_id=separator_id,
                                            max_token_length=max_seq_len, swap_source_target=swap_source_target)
test_dataset = ParaphraseDataset(dataset_conf, tokenizer, is_valid=True, lowercase=lower, no_special_token=no_special_token, 
                                            speaker_1_id=speaker_1_id, speaker_2_id=speaker_2_id, separator_id=separator_id,
                                            max_token_length=max_seq_len, swap_source_target=swap_source_target)

train_loader = GenerationDataLoader(dataset=train_dataset, model_type=model_type, tokenizer=tokenizer, max_seq_len=max_seq_len, 
                                    batch_size=train_batch_size, src_lid_token_id=src_lid, tgt_lid_token_id=tgt_lid, num_workers=8, shuffle=True)  
valid_loader = GenerationDataLoader(dataset=valid_dataset, model_type=model_type, tokenizer=tokenizer, max_seq_len=max_seq_len, 
                                    batch_size=valid_batch_size, src_lid_token_id=src_lid, tgt_lid_token_id=tgt_lid, num_workers=8, shuffle=False)
test_loader = GenerationDataLoader(dataset=test_dataset, model_type=model_type, tokenizer=tokenizer, max_seq_len=max_seq_len, 
                                   batch_size=test_batch_size, src_lid_token_id=src_lid, tgt_lid_token_id=tgt_lid, num_workers=8, shuffle=False)

n_epochs = 50
# n_epochs = 3 # filtered_paracotta
# n_epochs = 1 # full_paracotta

train(model, train_loader=train_loader, valid_loader=valid_loader, optimizer=optimizer, 
      forward_fn=forward_generation, metrics_fn=generation_metrics_fn, valid_criterion=valid_criterion, 
      tokenizer=tokenizer, n_epochs=n_epochs, evaluate_every=1, early_stop=early_stop, 
      grad_accum=grad_accumulate, step_size=step_size, gamma=gamma, 
      max_norm=max_norm, model_type=model_type, beam_size=beam_size,
      max_seq_len=max_seq_len, model_dir=model_dir, exp_id=0, fp16="", device=device)

# Load best model
model.load_state_dict(torch.load(model_dir + "/best_model_0.th"))

# Evaluate
test_loss, test_metrics, test_hyp, test_label = evaluate(model, data_loader=test_loader, forward_fn=forward_generation, 
                                                         metrics_fn=generation_metrics_fn, model_type=model_type, 
                                                         tokenizer=tokenizer, beam_size=beam_size, 
                                                         max_seq_len=max_seq_len, is_test=True, 
                                                         device='cuda')

metrics_scores = []
result_dfs = []

metrics_scores.append(test_metrics)
result_dfs.append(pd.DataFrame({
    'hyp': test_hyp, 
    'label': test_label
}))

result_df = pd.concat(result_dfs)
metric_df = pd.DataFrame.from_records(metrics_scores)

print('== Prediction Result ==')
print(result_df.head())
print()

print('== Model Performance ==')
print(metric_df.describe())

result_df.to_csv(model_dir + "/prediction_result.csv")
metric_df.describe().to_csv(model_dir + "/evaluation_result.csv")