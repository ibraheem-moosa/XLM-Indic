from datasets import set_caching_enabled, load_dataset
from transformers import AlbertForMaskedLM, AlbertTokenizer, AlbertConfig
from indicnlp.normalize import indic_normalize
import torch
from torch.utils.data import Dataset
import numpy as np
import tqdm
import tqdm.notebook
from aksharamukha import transliterate
import unicodedata

langs = ['pa', 'hi', 'bn', 'or', 'as', 'gu', 'mr']
tokenizer = AlbertTokenizer('../transliteration-tokenizer.model', model_max_length=2**32, do_lower_case=False, keep_accents=True)
def normalize_text(lang, text):
    script = {'pa': 'Gurmukhi', 'hi': 'Devanagari', 'bn': 'Bengali', 'or': 'Oriya', 'as': 'Assamese', 'gu': 'Gujarati', 'mr': 'Devanagari'}[lang]
    text = unicodedata.normalize('NFKC', text)
    text = indic_normalize.IndicNormalizerFactory().get_normalizer(lang).normalize(text)
    text = transliterate.process(script, 'ISO', text)
    text = text.replace('\n', ' ').strip()
    return text

indic_glue_csqa = dict((lang, load_dataset('indic_glue', 'csqa.' + lang)) for lang in langs)

class ClozeStyleQA(Dataset):
    
    def __init__(self, hf_dataset, tokenizer, lang):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.lang = lang
    
    def __getitem__(self, idx):
        title = self.hf_dataset[idx]['title']
        question = title + ' ' + self.hf_dataset[idx]['question']
        orig_question, question = question, normalize_text(self.lang, question)
        answer = self.hf_dataset[idx]['answer']
        options = self.hf_dataset[idx]['options']
        answer = options.index(answer)
        orig_options, options = options, [normalize_text(self.lang, option) for option in options]
        option_tokens = [self.tokenizer.encode(option, add_special_tokens=False) for option in options]
        option_token_counts = [len(token) for token in option_tokens]
        questions = [question.replace('<MASK>', '[MASK]' * tc) for tc in option_token_counts]
        instance = self.tokenizer(questions, 
#                                   return_special_tokens_mask=True,
                                  max_length=2**32, truncation=True, padding='longest', return_tensors='pt')
        return instance, option_tokens, answer#, orig_question, orig_options
        
    def __len__(self):
        return len(self.hf_dataset)

test_dataset = dict((lang, ClozeStyleQA(ds['test'], tokenizer, lang)) for lang, ds in indic_glue_csqa.items())
device = torch.device('cuda')
model = AlbertForMaskedLM.from_pretrained('../transliteration-model-outputs/checkpoint-1000000/checkpoint-1000000', output_attentions=False).to(device)
def get_option_probs(pred_logits, masked_tokens, option_tokens):
    pred_probas = torch.nn.functional.softmax(pred_logits, dim=2)
    probas = []
    for pred_proba, masked_token, option_token in zip(pred_probas, masked_tokens, option_tokens):
        pred_proba = pred_proba[masked_token]
        proba = 1.0
        for i, tok in enumerate(option_token):
            proba *= pred_proba[i][tok].item()
        probas.append(proba)
    return probas

def evaluate(model, tokenizer, device, test_dataset):
    predictions = []
    labels = []
    with torch.no_grad():
        for sample in tqdm.notebook.tqdm(test_dataset):
            model_input, option_tokens, answer = sample
            input_ids = model_input['input_ids'].to(device)
            token_type_ids = model_input['token_type_ids'].to(device)
            attention_mask = model_input['attention_mask'].to(device)
            masked_tokens = input_ids == tokenizer.mask_token_id
            pred_logits = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).logits
            option_probas = get_option_probs(pred_logits, masked_tokens, option_tokens)
            prediction = np.argmax(option_probas)
            predictions.append(prediction)
            labels.append(answer)
    return predictions, labels

from sklearn.metrics import accuracy_score

for lang, ds in test_dataset.items():
    preds, labels = evaluate(model, tokenizer, device, ds)
    preds, labels = np.array(preds), np.array(labels)
    np.save('csqa_' + lang + '_predictions.npy', preds)
    acc = accuracy_score(labels, preds)
    print('Accuracy for CSQA in language {} is {}.'.format(lang, acc))
