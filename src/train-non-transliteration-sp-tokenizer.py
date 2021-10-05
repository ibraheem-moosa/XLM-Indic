from datasets import load_from_disk, set_caching_enabled
import unicodedata
import sentencepiece as spm
import tqdm
import tqdm.notebook
import gc

dataset = load_from_disk('oscar-indo-aryan-filter-mismatched')
with open('oscar-indo-aryan-filter-mismatched.txt', 'w') as f:
    for sample in tqdm.notebook.tqdm(dataset):
        text = sample['text']
        text = text.replace('\n', ' ')
        text = unicodedata.normalize('NFKC', text)
        f.write(text + '\n')
del dataset
gc.collect()

spm.SentencePieceTrainer.train('--input=/root/dataset.txt --model_type=unigram --model_prefix=non-transliteration-tokenizer --train_extremely_large_corpus=true --split_digits=true --vocab_size=50000 --input_sentence_size=420000 --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 --pad_piece=<pad> --unk_piece=<unk> --bos_piece=[CLS] --eos_piece=[SEP] --user_defined_symbols=[MASK]')
