from datasets import load_from_disk, set_caching_enabled, load_dataset
from transformers import AlbertForSequenceClassification, AlbertTokenizer, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments, set_seed, TrainerCallback
from indicnlp.normalize import indic_normalize
import torch
from torch.utils.data import Dataset, ConcatDataset
from collections import Counter
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import unicodedata

tokenizer = AlbertTokenizer('../transliteration-tokenizer.model', model_max_length=512, do_lower_case=False, keep_accents=True)

def normalize_text(lang, text):
    script = {'pa': 'Gurmukhi', 'hi': 'Devanagari', 'bn': 'Bengali', 'or': 'Oriya', 'as': 'Assamese', 'gu': 'Gujarati', 'mr': 'Devanagari'}[lang]
    text = unicodedata.normalize('NFKC', text)
    text = indic_normalize.IndicNormalizerFactory().get_normalizer(lang).normalize(text)
    text = transliterate.process(script, 'ISO', text)
    text = text.replace('\n', ' ').strip()
    return text

indic_glue_wstp = {'pa': load_dataset('indic_glue', 'wstp.pa'),
                   'hi': load_dataset('indic_glue', 'wstp.hi'),
                   'bn': load_dataset('indic_glue', 'wstp.bn'),
                   'or': load_dataset('indic_glue', 'wstp.or'),
                   'as': load_dataset('indic_glue', 'wstp.as'),
                   'gu': load_dataset('indic_glue', 'wstp.gu'),
                   'mr': load_dataset('indic_glue', 'wstp.mr'),
                  }

for lang, dataset in indic_glue_wstp.items():
    indic_glue_wstp[lang] = indic_glue_wstp[lang].map(lambda s: {'lang': lang})

class WSTP(Dataset):
    
    def __init__(self, hf_dataset, tokenizer, lang):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.lang = lang
    
    def __getitem__(self, idx):
        context = self.hf_dataset[idx // 4]['sectionText']
        context = normalize_text(self.lang, context)
        chosen_title = ['titleA', 'titleB', 'titleC', 'titleD'][idx % 4]
        title = self.hf_dataset[idx // 4][chosen_title]
        title = normalize_text(self.lang, title)
        item = self.tokenizer(context, title, max_length=512, truncation=True)
        if chosen_title == self.hf_dataset[idx // 4]['correctTitle']:
            label = 1
        else:
            label = 0
        item['label'] = label
        return item
        
    def __len__(self):
        return len(self.hf_dataset) * 4

train_dataset = ConcatDataset([WSTP(ds['train'], tokenizer, lang) for lang, ds in indic_glue_wstp.items()])
valid_dataset = ConcatDataset([WSTP(ds['validation'], tokenizer, lang) for lang, ds in indic_glue_wstp.items()])

data_collator = DataCollatorWithPadding(tokenizer, padding='longest', max_length=512, pad_to_multiple_of=128)

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def compute_metrics(pred):
    num_choices = 4 # we have four choices for this multiple choice task
    labels = pred.label_ids
    labels = labels.reshape((-1, num_choices)).argmax(-1) # reshape and select the choice index
    preds = pred.predictions[..., 1] - pred.predictions[..., 0] # extract the difference of logit for positive prediction ie 1 and negative prediction ie 0
    preds = preds.reshape((-1, num_choices)).argmax(-1) # the logit difference of original binary task is the logit for this multiclass task
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro') # using micro average since we want to aggregate TP, FP etc 
    acc = accuracy_score(labels, preds)                                                        # on all samples because this 4 classes are made up ie 
                                                                                               # ie class 4 on two samples have no connection
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

accuracies = []
f1s = []
preds = []
for seed in [101, 102, 103, 104, 105, 106, 107, 108, 109]:
    set_seed(seed)
    model = AlbertForSequenceClassification.from_pretrained('../transliteration-model-outputs/checkpoint-1000000/checkpoint-1000000', 
                                                            output_attentions=False, num_labels=num_classes)
    training_args = TrainingArguments(
        output_dir='./results-{}'.format(seed),          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=32, #batch size per device during training
        per_device_eval_batch_size=32,# batch size for evaluation
        warmup_ratio=0.10,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        learning_rate=2e-5,
        logging_dir='./logs-{}'.format(seed),            # directory for storing logs
        logging_steps=1,
        save_strategy='no',
        logging_strategy='epoch',
        report_to='tensorboard',
        load_best_model_at_end=False,
    #     metric_for_best_model='f1',
    #     greater_is_better=True,
        do_train=True,
        do_eval=True,
        evaluation_strategy='epoch',
        tpu_num_cores=8,
        dataloader_num_workers=4,
    #     eval_accumulation_steps=1
    ) 

    trainer = Trainer(model=model, args=training_args, 
                  data_collator=data_collator, 
                  train_dataset=train_dataset, 
                  eval_dataset=valid_dataset, 
                  compute_metrics=compute_metrics,
                  tokenizer=tokenizer)

    trainer.train()

    for lang in indic_glue_wstp.keys():
        print('Evaluating test dataset for lang: {}'.format(lang))
        test_dataset = WSTP(indic_glue_wstp[lang]['test'], tokenizer, lang)
        predictions, labels, metric = trainer.predict(test_dataset)
        print(metric)
