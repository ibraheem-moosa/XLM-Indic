from datasets import load_from_disk, set_caching_enabled, load_dataset
from transformers import AlbertTokenizer, DataCollatorForTokenClassification
from transformers import Trainer, TrainingArguments, set_seed, EvalPrediction
from indicnlp.normalize import indic_normalize
import torch
from torch.utils.data import Dataset, ConcatDataset
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import unicodedata
from aksharamukha import transliterate


indic_glue_ner = {'pa': load_dataset('wikiann', 'pa'),
                  'hi': load_dataset('wikiann', 'hi'),
                  'bn': load_dataset('wikiann', 'bn'),
                  'or': load_dataset('wikiann', 'or'),
                  'as': load_dataset('wikiann', 'as'),
                  'gu': load_dataset('wikiann', 'gu'),
                  'mr': load_dataset('wikiann', 'mr'),
                 }

for lang, dataset in indic_glue_ner.items():
    indic_glue_ner[lang] = indic_glue_ner[lang].map(lambda s: {'lang': lang})

ner_tag_id_to_name = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'I-LOC', 6: 'B-LOC', 7: 'C'}
label_names = [ner_tag_id_to_name[i] for i in range(len(ner_tag_id_to_name))]
ner_tag_name_to_id = dict(map(reversed, ner_tag_id_to_name.items()))
label_counter = Counter()
for lang, ds in indic_glue_ner.items():
    for ner_tags in ds['train']['ner_tags']:
        label_counter.update(ner_tags)
for key, value in label_counter.most_common():
    print('{}: {}'.format(ner_tag_id_to_name[key], value / sum(label_counter.values())))
with torch.no_grad():
    label_freqs = torch.tensor([label_counter[key] for key in range(7)]) / sum(label_counter.values())


tokenizer = AlbertTokenizer('../transliteration-tokenizer.model', model_max_length=512, do_lower_case=False, keep_accents=True)
def normalize_text(lang, text):
    script = {'pa': 'Gurmukhi', 'hi': 'Devanagari', 'bn': 'Bengali', 'or': 'Oriya', 'as': 'Assamese', 'gu': 'Gujarati', 'mr': 'Devanagari'}[lang]
    text = unicodedata.normalize('NFKC', text)
    text = indic_normalize.IndicNormalizerFactory().get_normalizer(lang).normalize(text)
    text = transliterate.process(script, 'ISO', text)
    text = text.replace('\n', ' ').strip()
    return text

class WikiAnnNER(Dataset):
    
    def __init__(self, hf_dataset, tokenizer, lang):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.lang = lang
    
    def __getitem__(self, idx):
#        lang = self.hf_dataset[idx]['lang']
        original_tokens = self.hf_dataset[idx]['tokens']
        original_tokens = [normalize_text(self.lang,token) for token in original_tokens]
        token_lengths = [len(self.tokenizer.encode(token)) - 2 for token in original_tokens] # We will use this to generate ner_tags as neccessary
        ner_tags = self.hf_dataset[idx]['ner_tags']
        generated_ner_tags = [ner_tag_name_to_id['O']]
        for ner_tag, token_length in zip(ner_tags, token_lengths):
            generated_ner_tags.append(ner_tag)
            generated_ner_tags.extend([ner_tag_name_to_id['C']] * (token_length - 1))
        generated_ner_tags.append(ner_tag_name_to_id['O'])
        item = self.tokenizer(original_tokens, max_length=512, truncation=True, is_split_into_words=True)#, padding='max_length')
        label = generated_ner_tags# + [ner_tag_name_to_id['C']] * (512 - len(generated_ner_tags))
#         print(np.array(label).shape, np.array(item['input_ids']).shape)
        assert(len(item['input_ids']) == len(label))
        item['labels'] = label
#         print(item['input_ids'], item['label'])
        return item
        
    def __len__(self):
        return len(self.hf_dataset)

train_dataset = ConcatDataset([WikiAnnNER(ds['train'], tokenizer, lang) for lang, ds in indic_glue_ner.items()])
valid_dataset = ConcatDataset([WikiAnnNER(ds['validation'], tokenizer, lang) for lang, ds in indic_glue_ner.items()])

label_counter = Counter()
for sample in train_dataset:
    label_counter.update(sample['labels'])
for key, value in label_counter.most_common():
    print('{}: {}'.format(ner_tag_id_to_name[key], value / sum(label_counter.values())))
with torch.no_grad():
    label_freqs = torch.tensor([label_counter[key] for key in range(8)]) / sum(label_counter.values())
label_freqs

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def compute_metrics(pred):
    labels = pred.label_ids.flatten()
    preds = pred.predictions.argmax(-1).flatten()
    # remove padding labels
    valid_label_indices = np.nonzero(labels != -100)
#     print(labels.shape, preds.shape)
    labels = labels[valid_label_indices]
    preds = preds[valid_label_indices]
#     print(labels.shape, preds.shape)
    # ignore continution label C when calculating metric
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', labels=list(range(7)))
    cm = confusion_matrix(labels, preds, labels=list(range(7)))
    # calculate accuracy from confusion matrix
    acc = np.trace(cm) / np.sum(cm) #accuracy_score(labels, preds)
#     print(cm)
    cm_disp = ConfusionMatrixDisplay(cm, display_labels=label_names[:7])
    cm_disp.plot()
    plt.show()
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

data_collator = DataCollatorForTokenClassification(tokenizer, padding='longest', max_length=512, pad_to_multiple_of=16)

accuracies = defaultdict(list)
f1s = defaultdict(list)
preds = defaultdict(list)
for seed in [101, 102, 103, 104, 105, 106, 107, 108, 109]:
    set_seed(seed)
    model = AlbertForTokenClassification.from_pretrained('../transliteration-model-outputs/checkpoint-1000000/checkpoint-1000000', 
                                                            output_attentions=False, num_labels=len(ner_tag_id_to_name))
    with torch.no_grad():
        model.classifier.bias.copy_(torch.log(label_freqs))
        
    training_args = TrainingArguments(
        output_dir='./results-{}'.format(seed),          # output directory
        num_train_epochs=20,              # total number of training epochs
        per_device_train_batch_size=64, #batch size per device during training
        per_device_eval_batch_size=128,# batch size for evaluation
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
        dataloader_num_workers=4,
        tpu_num_cores=8,
    #     eval_accumulation_steps=1
    )
    
    trainer = Trainer(model=model, args=training_args, 
                  data_collator=data_collator, 
                  train_dataset=train_dataset,
                  eval_dataset=valid_dataset, 
                  compute_metrics=compute_metrics,
                  tokenizer=tokenizer)
    
    trainer.train()
    
    for lang in indic_glue_ner.keys():
        print('Evaluating test dataset for lang: {}'.format(lang))
        test_dataset = WikiAnnNER(indic_glue_ner[lang]['test'], tokenizer, lang)
        predictions, labels, metric = trainer.predict(test_dataset)
        print('Metrics for LANG {} SEED {}: Acc: {} F1:{}'.format(lang, seed, metric['test_accuracy'], metric['test_f1']))
        
        accuracies[lang].append(metric['test_accuracy'])
        f1s[lang].append(metric['test_f1'])
        preds[lang].append(predictions)


for lang in indic_glue_ner.keys():
    print('Metric STATS LANG {} Avg Acc: {} Std Acc{}: Avg F1: {} Std F1: {}'.format(lang, np.mean(accuracies[lang]), np.std(accuracies[lang]), np.mean(f1s[lang]), np.std(f1s[lang])))
