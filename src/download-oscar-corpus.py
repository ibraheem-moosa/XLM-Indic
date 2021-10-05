from datasets import load_dataset, load_from_disk, concatenate_datasets
from aksharamukha import transliterate

dataset_as = load_dataset("oscar", "unshuffled_deduplicated_as", split='train')
dataset_or = load_dataset("oscar", "unshuffled_deduplicated_or", split='train')
dataset_bh = load_dataset("oscar", "unshuffled_deduplicated_bh", split='train')
dataset_ne = load_dataset("oscar", "unshuffled_deduplicated_ne", split='train')
dataset_bpy = load_dataset("oscar", "unshuffled_deduplicated_bpy", split='train')
dataset_sa = load_dataset("oscar", "unshuffled_deduplicated_sa", split='train')
dataset_mai = load_dataset("oscar", "unshuffled_deduplicated_mai", split='train')
dataset_bn = load_dataset("oscar", "unshuffled_deduplicated_bn", split='train')
dataset_gu = load_dataset("oscar", "unshuffled_deduplicated_gu", split='train')
dataset_pa = load_dataset("oscar", "unshuffled_deduplicated_pa", split='train')
dataset_gom = load_dataset("oscar", "unshuffled_deduplicated_gom", split='train')
dataset_mr = load_dataset("oscar", "unshuffled_deduplicated_mr", split='train')
dataset_si = load_dataset("oscar", "unshuffled_deduplicated_si", split='train')
dataset_hi = load_dataset("oscar", "unshuffled_deduplicated_hi", split='train')

dataset_as = dataset_as.map(lambda x: {"text": x["text"], "id": f"as[{x['id']}]"})
dataset_or = dataset_or.map(lambda x: {"text": x["text"], "id": f"or[{x['id']}]"})
dataset_bh = dataset_bh.map(lambda x: {"text": x["text"], "id": f"bh[{x['id']}]"})
dataset_ne = dataset_ne.map(lambda x: {"text": x["text"], "id": f"ne[{x['id']}]"})
dataset_bpy = dataset_bpy.map(lambda x: {"text": x["text"], "id": f"bpy[{x['id']}]"})
dataset_sa = dataset_sa.map(lambda x: {"text": x["text"], "id": f"sa[{x['id']}]"})
dataset_mai = dataset_mai.map(lambda x: {"text": x["text"], "id": f"mai[{x['id']}]"})
dataset_bn = dataset_bn.map(lambda x: {"text": x["text"], "id": f"bn[{x['id']}]"})
dataset_gu = dataset_gu.map(lambda x: {"text": x["text"], "id": f"gu[{x['id']}]"})
dataset_pa = dataset_pa.map(lambda x: {"text": x["text"], "id": f"pa[{x['id']}]"})
dataset_gom = dataset_gom.map(lambda x: {"text": x["text"], "id": f"gom[{x['id']}]"})
dataset_mr = dataset_mr.map(lambda x: {"text": x["text"], "id": f"mr[{x['id']}]"})
dataset_si = dataset_si.map(lambda x: {"text": x["text"], "id": f"si[{x['id']}]"})
dataset_hi = dataset_hi.map(lambda x: {"text": x["text"], "id": f"hi[{x['id']}]"})

dataset = concatenate_datasets([dataset_as, dataset_or, dataset_bh, dataset_ne, dataset_bpy, dataset_sa, dataset_mai, dataset_bn, dataset_gu, dataset_pa, dataset_gom, dataset_mr, dataset_si, dataset_hi])

def script_matches_label(sample):
    lang = sample['id'].split('[')[0]
    text = sample['text']
    lang_script = {'as': 'Bengali', 'or': 'Oriya', 'bn': 'Bengali', 'pa': 'Gurmukhi', 'gu': 'Gujarati', 'si': 'Sinhala', 
                        'bpy': 'Bengali',
                        'bh': 'Devanagari', 'mr': 'Devanagari', 'hi': 'Devanagari', 'gom': 'Devanagari', 'mai': 'Devanagari', 'sa': 'Devanagari', 'ne': 'Devanagari'}
    try:
        detected_script = transliterate.auto_detect(text)
    except (IndexError, ValueError, KeyError) as e:
        print(e, text)
        return False
    if detected_script == 'Assamese':
        detected_script = 'Bengali'
    return lang_script[lang] == detected_script

dataset = dataset.filter(script_matches_label, num_proc=4)
dataset.save_to_disk('oscaar-indo-aryan-filter-mismatched')
