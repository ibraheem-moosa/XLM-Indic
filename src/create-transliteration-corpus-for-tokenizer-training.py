from datasets import load_from_disk, set_caching_enabled
from indicnlp.tokenize import sentence_tokenize
from indicnlp.normalize import indic_normalize
from aksharamukha import transliterate

dataset = load_from_disk('oscar-indo-aryan-filter-mismatched')

def transliterate_to_iso(lang, text):
    lang_script = {'as': 'Assamese', 'or': 'Oriya', 'bn': 'Bengali', 'pa': 'Gurmukhi', 'gu': 'Gujarati', 'si': 'Sinhala', 
                        'bpy': 'Assamese',
                        'bh': 'Devanagari', 'mr': 'Devanagari', 'hi': 'Devanagari', 'gom': 'Devanagari', 'mai': 'Devanagari', 'sa': 'Devanagari', 'ne': 'Devanagari'}
    text = transliterate.process(lang_script[lang], 'ISO', text)
    return text

def split_into_sentences(sample):
    lang = sample['id'].split('[')[0]
    if lang == 'mai':
        lang = 'hi'
    elif lang == 'gom':
        lang = 'kK'
    elif lang == 'bpy':
        lang = 'as'
    elif lang == 'bh':
        lang = 'hi'
    text = sample['text']
    text = indic_normalize.IndicNormalizerFactory().get_normalizer(lang).normalize(text)
#     if lang == 'as' or lang == 'bn':
#         delim_pat = re.compile(r'[|?!।॥]')
#     else:
#         delim_pat = re.compile(r'[|.?!।॥]')
    sentences = sentence_tokenize.sentence_split(text, lang)#, delim_pat=delim_pat)
    sentences = '[SEP]'.join(sentences)
    return {'id': sample['id'], 'text': sentences}

def transliterate_sentences(sample):
    lang = sample['id'].split('[')[0]
#     sentences = list(map(lambda s: transliterate_to_iso(lang, s), sample['text']))
    sentences = transliterate_to_iso(lang, sample['text'])
    return {'id': sample['id'], 'text': sentences}

def split_and_transliterate(sample):
    return transliterate_sentences(split_into_sentences(sample))

dataset = dataset.map(split_and_transliterate, num_proc=4)

dataset.save_to_disk('oscaar-indo-aryan-split-transliterated')
