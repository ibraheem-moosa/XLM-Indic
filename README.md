# Introduction

This is the source code for our paper [Transliteration: A Simple Technique For Improving Multilingual Language Modeling](https://openreview.net/forum?id=NqDLrS73nG).
We study how to improve the performance of Multilingual Masked Language Models (MLLM) on closely related languages that use different writing scripts. We find that using off-the-shelf
standard rule-based transliteration as a pre-processing step, we can significantly improve performance especially in low-resource languages. 

To motivate this, consider the following situations.
1. Serbian and Croatian are so closely related that some linguists consider them to be the same language. However Serbian and Croatian is written in Cyrillic and Latin script respectively.
2. Similar situation is present for Hindi and Urdu. Hindi is written using Devanagari script while Urdu is written in Perso-Arabic script.
3. Many Turkic languages (Kazakh, Uzbek, Turkmen etc) used to be written in Cyrillic scripts but now officially use Latin script. Depending on the region the same language is written in different scripts.
4. Bengali, Hindi, Urdu, Gujarati, Punjabi, Oriya, Divehi and Sinhala are closely related related languages from the Indo-Aryan language family. However each of these languages are written in a separate script.
5. Same situation is present for the Dravidian languages Tamil, Malayalam, Kannada and Telugu.
6. Two scripts (Traditional and Simplified) are officially used to write Mandarin.
7. Thai and Lao are two closely related languages that use different scripts.
8. Many scientific words in English and Russian are borrowed/originally from Greek. For example, Biology and биология are from βιολογία. While we humans know this a language model only sees three different words in three different scripts.

The examples can go on and on. There are many different scripts being used all over the world and in many situations they present a barrier to sharing multilingual represenataions across languages. Now the question is can we do something about this and improve the performance of MLLMs? The answer is YES!

# Transliteration
There are ISO standards that lay down the rules for converting text written in one script from another. For example, [ISO-9](https://en.wikipedia.org/wiki/ISO_9) is a standard for transliterating Cyrillic to Latin. [ISO-843](https://en.wikipedia.org/wiki/ISO_843) transliterates from Greek script to Latin script. [ISO-15919](https://en.wikipedia.org/wiki/ISO_15919) is a standard for transliterating many Indian scripts to Latin script. See a list of avaialble ISO transliteration scheme [here](https://en.wikipedia.org/wiki/List_of_ISO_romanizations).

But do we have standard implementations for them and do they work well? The answer to both of these question is yes. The [PyICU](https://pypi.org/project/PyICU/) library provides standard implementations for these transliteration schemes. And you can test how well the transliteration works using this [demo](https://icu4c-demos.unicode.org/icu-bin/translit). For example, when we put биология and βιολογία there we get biologiâ and biología respectively.

It is intuitively clear that if we transliterate texts in different scripts to a common scirpt (ie Latin), our trained MLLM may be better able to link words and concepts across languages. We also have a working library to transliterate across many different scripts. Now we need to test whether transliterating to the same script actually helps MLLMs or not.

# Testing Our Hypothesis
Now that we have found a library for reliably transliterating scripts we can formulate experiments to test our hypothesis. We need evaluation datasets on which we can test our models and we need this evaluation dataset on languages that are closely related but use different scripts.

The [IndicGLUE](https://indicnlp.ai4bharat.org/indic-glue/) benchmark dataset is almost perfect for testing our hypothesis. This dataset spans 11 Indian languages using nine different scripts. Four of these languages are from the Dravidian language family and the rest are from the Indo-Aryan language family. These nine scripts can be transliterated to Latin using the [ISO-15919](https://en.wikipedia.org/wiki/ISO_15919) transliteration scheme.

To reduce the scope due to our compute constraints (and further reasons explained in our paper) we limit ourselves to just the Indo-Aryan porion of the dataset. We pre-train one model with the original scripts and another after transliterating the text with ISO-15919 transliteration scheme. We pre-train on the OSCAR corpus.

We evaluate the two models on the IndicGLUE benchmark dataset. You can find the detailed results in our paper, but here is the summary.

1. The transliteration based model performs significantly better on low resource languages.
2. On some tasks in the high resource languages it also performs significantly better.

This indicates that the transliteration based model has learned better representations. It is most likely using the representations learned on high resource languages to perform better on low resource languages. To get a direct sense of whether this is true we calculated the similarity of internal representations of the models across parallel sentences from different languages. We use the [FLORES-101](https://arxiv.org/abs/2106.03193) dataset from Facebook that was released last year. We find that the internal representations of the transliteration model are indeed more similar across different languages. See the appendix of our paper.

# Reproducing Our Results

We have provided all the scripts to reporduce our main results.
Use the provided script to do the following in this order.
1. Download oscar corpus.
2. Train non transliteration sentence piece tokenizer.
3. Create corpus for training non transliteration model.
4. Pretrain the non transliteration model.
5. Create transliteration corpus for sentence piece tokenzier training.
6. Train transliteration sentence piece tokenizer.
7. Create corpus for training transliteration model.
8. Pretrain the transliteration model.
9. Run the fine tuning scirpts for the non transliteration model.
10. Run the fine tuning scripts for the transliteration model.

We will add the cross-lingual representation similarity scripts soon.

# Other Projects We are Currently Working On

We are currently working on a few exciting projects. We are looking for collaborators who can help us with compute resources and ideas as well as data processing and model implementation. Feel free to send an email.

1. Does our finding hold for larger models? It is possible that large models already learn these associations across different scripts from the few pivots available in the pretraining corpus. Is the improvement still sginificant for large models? We are currently pretaining large models on a larger set of languages. Pretraining is almost done. We will probably get the benchmark results in a couple of weeks (around Jan 21).
2. It is possible that by transliterating we are sharing too much tokens across languages which may confuse the model. It is easy to see that this can happen if the languages in question are unrelated but it is also possible even if the languages are related. We are currently working on experiment design to test this. Feel free to email us if you are interested.
3. In our paper we worked with sentencepiece tokenization. What happens if we use character based tokenization. We have a hypothesis that the difference in performance would increase for a character based model. This project is also in experiment design phase. Feel free to contact with us if you are interested.
4. Did you see the recent [paper](https://nips.cc/Conferences/2021/ScheduleMultitrack?event=27434) from Deepmind that shows that LMs generalaization can decrease with time. We have a hypotheis that character based models can do better temporal generalization. Character based models have been lagging behind subword based models due to the increased compute requirement. Maybe in terms of temporal genearalization we can get better performance. Also we are currently building a timestamped corpus for Bengali. This project is on a very preliminary stage. Any help/collaboration is highly appreciated.
5. We are also working on a translation of WNLI to Bengali which is almost done.
6. Also look at our submitted [idea](https://docs.google.com/document/d/1seolfb0yeW6m1cpn7ej3Qx-5DGegwFliTJZ8nbW6wSk/) on ICLR CSS.
