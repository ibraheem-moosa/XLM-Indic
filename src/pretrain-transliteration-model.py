from datasets import load_from_disk, set_caching_enabled
import torch
from torch.utils.data import Dataset
import regex as re
import random
from transformers import AlbertForPreTraining, AlbertConfig, AlbertTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, TrainerCallback

set_caching_enabled(False)
tokenizer = AlbertTokenizer('transliteration-tokenizer.model', model_max_length=512, do_lower_case=False, keep_accents=True)
dataset = load_from_disk('transliteration-split-merge-sentences')
dataset = dataset['train']

class LMDataset(Dataset):
    
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.word_splitter_pattern = re.compile(r'[\p{Separator}\p{Punctuation}\p{Other}\p{Symbol}\p{Number}]')
    
    def __getitem__(self, idx):
        line = self.dataset[idx]['text']
        split_points = [m.start() for  m in self.word_splitter_pattern.finditer(line)]
        # randomly choose a split point
        try:
            chosen_split_point = random.choice(split_points)
        except IndexError:
            chosen_split_point = len(line) // 2
        line_A, line_B = line[:chosen_split_point], line[chosen_split_point:]
        # randomly swap A and B with 50pc chance
        is_random_next = random.random() < 0.5
        if is_random_next:
            instance = {'input_ids': self.tokenizer.encode(line_B, line_A, return_special_tokens_mask=True, max_length=512, truncation=True, padding='longest')}
        else:
            instance = {'input_ids': self.tokenizer.encode(line_A, line_B, return_special_tokens_mask=True, max_length=512, truncation=True, padding='longest')}
#         print(instance)
        instance['sentence_order_label'] = 1 if is_random_next else 0
        return instance
    
    def __len__(self):
        return len(self.dataset)

dataset = LMDataset(dataset, tokenizer)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, pad_to_multiple_of=8)
config = AlbertConfig(vocab_size=50000, embedding_size=128, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, max_position_embeddings=512)
model = AlbertForPreTraining(config)
training_args = TrainingArguments(output_dir='transliteration-model-outputs/', do_train=True, per_device_train_batch_size=32, per_device_eval_batch_size=32, 
                                  learning_rate=1e-3/8, weight_decay=1e-2, adam_epsilon=1e-6, max_grad_norm=1.0, max_steps=1000000, warmup_steps=5000, 
                                  logging_steps=1000, save_steps=10000, dataloader_num_workers=4, report_to="none", dataloader_drop_last=True,
                                  ignore_data_skip=True,
                                  seed=1, 
                                  tpu_num_cores=8
                                )
trainer = Trainer(model=model, args=training_args, data_collator=data_collator, train_dataset=dataset)
trainer.train()
