from datasets import load_from_disk, set_caching_enabled, load_dataset
from indicnlp.tokenize import sentence_tokenize
from indicnlp.normalize import indic_normalize
import unicodedata
import sentencepiece as spm
import tqdm
import tqdm.notebook

sp = spm.SentencePieceProcessor('non-transliteration-tokenizer.model')
dataset = load_from_disk('oscar-indo-aryan-filter-mismatched')

def split_into_sentences(lang, text):
    if lang == 'mai':
        lang = 'hi'
    elif lang == 'gom':
        lang = 'kK'
    elif lang == 'bpy':
        lang = 'as'
    elif lang == 'bh':
        lang = 'hi'
    text = unicodedata.normalize('NFKC', text)
    text = indic_normalize.IndicNormalizerFactory().get_normalizer(lang).normalize(text)
    sentences = sentence_tokenize.sentence_split(text, lang)
    return sentences

def get_merges(word_counts, max_length=512):
    curr_length = word_counts[0]
    merge_starts = []
    for i, wc in enumerate(word_counts[1:]):
        if curr_length + wc <= max_length:
            curr_length += wc
        else:
            merge_starts.append(i + 1)
            curr_length = wc
    return merge_starts

def merge_sentences(sample):
    lang = sample['id'].split('[')[0]
    sentences = split_into_sentences(lang, sample['text'])
    word_counts = list(map(lambda s: len(sp.encode(s)), sentences))
    merge_starts = get_merges(word_counts, max_length=512) #keep three tokens for CLS, SEP, SEP in later version
    merged_sentences = []
    current_sentence = []
    for i, sentence in enumerate(sentences):
        if i in merge_starts:
            merged_sentences.append(' '.join(current_sentence))
            current_sentence.clear()
        current_sentence.append(sentence)
    merged_sentences.append(' '.join(current_sentence))
    return {'id': sample['id'], 'text': merged_sentences}

dataset = dataset.map(merge_sentences, num_proc=4)

with open('no-transliteration-split-merge-sentences.txt', 'w') as f:
    for sample in tqdm.notebook.tqdm(dataset):
        for sentence in sample['text']:
            f.write(sentence.replace('\n', ' ') + '\n')

dataset = load_dataset('text', data_files='no-transliteration-split-merge-sentences.txt')
dataset = dataset.filter(lambda s: len(s['text']) > 512, num_proc=4)
dataset = dataset.map(lambda sample: {'length': len(sp.encode(sample['text']))}, num_proc=4)
dataset.save_to_disk('no-transliteration-split-merge-sentences')
