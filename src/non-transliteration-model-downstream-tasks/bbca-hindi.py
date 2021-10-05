from datasets import load_from_disk, set_caching_enabled, load_dataset
from transformers import AlbertForSequenceClassification, AlbertTokenizer, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments, set_seed, EvalPrediction
from indicnlp.normalize import indic_normalize
import torch
from torch.utils.data import Dataset
from collections import Counter
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import unicodedata
from aksharamukha import transliterate

tokenizer = AlbertTokenizer('../non-transliteration-tokenizer.model', model_max_length=512, do_lower_case=False, keep_accents=True)
data_collator = DataCollatorWithPadding(tokenizer, padding='longest', max_length=512)
lang = 'hi'
def normalize_text(text):
    script = {'pa': 'Gurmukhi', 'hi': 'Devanagari', 'bn': 'Bengali', 'or': 'Oriya', 'as': 'Assamese', 'gu': 'Gujarati', 'mr': 'Devanagari'}[lang]
    text = unicodedata.normalize('NFKC', text)
    text = indic_normalize.IndicNormalizerFactory().get_normalizer(lang).normalize(text)
    text = text.replace('\n', ' ').strip()
    return text

indic_glue_bbca = load_dataset('indic_glue', 'bbca.hi')
labels = indic_glue_bbca['train']['label']
indices = list(range(len(labels)))
train_indices, valid_indices, train_labels, valid_labels = train_test_split(indices, labels, train_size=0.70, stratify=labels, random_state=1)
indic_glue_bbca_train_validation_split = {'train': indic_glue_bbca['train'].select(train_indices), 'test': indic_glue_bbca['train'].select(valid_indices)}

labels = indic_glue_bbca['train'].unique('label')
num_classes = len(labels)
labels_dict = dict((name, index) for index, name in enumerate(labels))

class BBCHindiNewsClassification(Dataset):
    
    def __init__(self, hf_dataset, tokenizer, split, labels_dict):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.split = split
        self.labels_dict = labels_dict
    
    def __getitem__(self, idx):
        text = self.hf_dataset[self.split][idx]['text']
        text = normalize_text(text)
        item = self.tokenizer(text, max_length=512, truncation=True, padding='longest')
        label = self.hf_dataset[self.split][idx]['label']
        item['label'] = self.labels_dict[label]
        return item
        
    def __len__(self):
        return len(self.hf_dataset[self.split])



train_dataset = BBCHindiNewsClassification(indic_glue_bbca_train_validation_split, tokenizer, 'train', labels_dict)
valid_dataset = BBCHindiNewsClassification(indic_glue_bbca_train_validation_split, tokenizer, 'test', labels_dict)

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)
#     print(cm)
    cm_disp = ConfusionMatrixDisplay(cm)
    cm_disp.plot()
    plt.show()
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

from scipy.special import softmax

def get_ensemble_predictions(predictions):
    predictions = np.stack(predictions)
    predictions = softmax(predictions, axis=-1).mean(axis=0)
    return predictions

accuracies = []
f1s = []
preds = []
for seed in [101, 102, 103, 104, 105, 106, 107, 108, 109]:
    set_seed(seed)
    model = AlbertForSequenceClassification.from_pretrained('../non-transliteration-model-outputs/checkpoint-1000000/checkpoint-1000000', 
                                                            output_attentions=False, num_labels=num_classes)
    training_args = TrainingArguments(
        output_dir='./results-{}'.format(seed),          # output directory
        num_train_epochs=20,              # total number of training epochs
        per_device_train_batch_size=16, #batch size per device during training
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
        dataloader_num_workers=2,
    #     eval_accumulation_steps=1
    )
    
    trainer = Trainer(model=model, args=training_args, 
                  data_collator=data_collator, 
                  train_dataset=train_dataset,
                  eval_dataset=valid_dataset, 
                  compute_metrics=compute_metrics,
                  tokenizer=tokenizer)
    
    trainer.train()
    
    test_dataset = BBCHindiNewsClassification(indic_glue_bbca, tokenizer, 'test', labels_dict)
    predictions, labels, metric = trainer.predict(test_dataset)
    print('Metrics for SEED {}: Acc: {} F1:{}'.format(seed, metric['test_accuracy'], metric['test_f1']))
    
    accuracies.append(metric['test_accuracy'])
    f1s.append(metric['test_f1'])
    preds.append(predictions)

print('Metric STATS Avg Acc: {} Std Acc{}: Avg F1: {} Std F1: {}'.format(np.mean(accuracies), np.std(accuracies), np.mean(f1s), np.std(f1s)))

ensemble_predictions = get_ensemble_predictions(preds)
ensemble_metric = compute_metrics(EvalPrediction(ensemble_predictions, labels))
print('Metric for ENSEMBLE Acc: {} F1: {}'.format(ensemble_metric['accuracy'], ensemble_metric['f1']))
