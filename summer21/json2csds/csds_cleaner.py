import re
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

import json
from json2csds import JSON2CSDS

from nltk.tokenize.treebank import TreebankWordDetokenizer
detokenizer = TreebankWordDetokenizer()


def alert_wrong_anno(anno, doc_id, error=None):
    """
    It is used for alerting wrong annotation(s).
    :param anno: The annotation that error(s) were happening in its process.
    :param error: The error(s) that happened.
    """

    if str(error) != str("'text'"):
        print('===================\nWrong Clean!!')
        print(anno)
        print('Error details: (doc_id: ', doc_id, ')')
        print(error)
        print(f'Type of error: {error.__class__.__name__}')
        print('===================')


def clean_item(txt):
    re_pattern = '[a-zA-Z0-9 _=+/\"\'\-]'

    txt = re.sub('\n', '  ', txt)
    txt = re.sub('<UH>', 'UUHH', txt)
    txt = re.sub('<' + re_pattern + '*>', '', txt)
    txt = re.sub(re_pattern + '+>', '', txt)
    txt = re.sub('--', ' -- ', txt)
    txt = re.sub('  ', ' ', txt)

    indices = []
    for i in range(len(txt)):
        if ord(txt[i]) == 96: # or ord(txt[i]) == 39:
            indices.append(i)

    if len(indices) > 0:
        i = 0
        while i < len(indices):
            txt = txt[0: indices[i]] + chr(34) + txt[indices[i] + 1:]
            i += 1


    return txt


def back_to_clean(lst):
    txt = detokenizer.detokenize(lst)
    txt = re.sub(' \.', '.', txt)
    return txt


def char_to_word(item_id="", text="", head="", start=0, end=0, clean=False, verbose=False):
    text1 = text[0: start]
    text2 = text[start: end]
    text3 = text[end:]

    if clean:
        text_tokens1 = word_tokenize(clean_item(text1))
        text_tokens2 = word_tokenize(clean_item(text2))
        text_tokens3 = word_tokenize(clean_item(text3))
        all_text_tokens = word_tokenize(clean_item(text))

        text_tokens1 = list(map(clean_item, text_tokens1))
        text_tokens2 = list(map(clean_item, text_tokens2))
        text_tokens3 = list(map(clean_item, text_tokens3))
        all_text_tokens = list(map(clean_item, all_text_tokens))
    else:
        text_tokens1 = word_tokenize(text1)
        text_tokens2 = word_tokenize(text2)
        text_tokens3 = word_tokenize(text3)
        all_text_tokens = word_tokenize(text)

    if verbose and all_text_tokens != text_tokens1 + text_tokens2 + text_tokens3:
        print(f" <Warning word tokenization mismatch id=[{item_id}]: head={text2} | w_head={text_tokens2} | text={all_text_tokens}/>")

    # returns start index, list of tokens and the length of the tokens after the first index which should be considered
    return {
        'w_head_span': (len(text_tokens1), len(text_tokens1)+len(text_tokens2)),
        'w_text': all_text_tokens,
        'w_head': text_tokens2,
        'clean_text': back_to_clean(all_text_tokens),
        'clean_head': back_to_clean(text_tokens2)
    }


def find_info(ids, data_subset, clean=False, add_attitude_attributes=False, parent_w_text=[], verbose=False, data_targets={}):
    word_based_info = {}
    word_based_info_list = []
    if ids:
        for item_id in ids:
            if type(data_subset) is dict:
                if item_id in data_subset:
                    item = data_subset[item_id]  #dictionary: char based for sentence, word_based for sentence array, aspect, polarity, intensity, type
                    word_based_info = char_to_word(
                        item_id=item_id, text=item['text'], head=item['head'], start=item['head_start'], end=item['head_end'], clean=clean, verbose=verbose
                    )
                    if add_attitude_attributes:
                        word_based_info.update({
                            'annotation_type': item['annotation_type'],
                            'polarity': item['polarity'],
                            'intensity': item['intensity'],
                        })
                    if verbose and parent_w_text != [] and word_based_info['w_text'] != [] and parent_w_text != word_based_info['w_text']:
                        print(f' <Error sentence mismatch id={item_id}: parent={parent_w_text} | child={word_based_info["w_text"]}')
                elif verbose:
                    print(f" <Warning id={item_id} couldn't be found./>")
            else:
                for item in data_subset:
                    if item_id == item['unique_id']:
                        word_based_info = char_to_word(
                            item_id=item_id, text=item['text'], head=item['head'], start=item['head_start'],
                            end=item['head_end'], clean=clean, verbose=verbose
                        )
                        if add_attitude_attributes:
                            word_based_info.update({
                                'annotation_type': item['annotation_type'],
                                'polarity': item['polarity'],
                                'intensity': item['intensity'],
                                'target': find_info(item['target_link'], data_targets, clean, parent_w_text=word_based_info['w_text'], verbose=verbose)
                            })
                        if verbose and parent_w_text != [] and word_based_info['w_text'] != [] and parent_w_text != \
                                word_based_info['w_text']:
                            print(
                                f' <Error sentence mismatch id={item_id}: parent={parent_w_text} | child={word_based_info["w_text"]}')

            word_based_info_list.append(word_based_info)
    # else:
        #print()

    return word_based_info_list

def tokenize_and_extract_info(data_address, save_address, clean=False, verbose=False, activate_progressbar=True):
    obj = JSON2CSDS("MPQA2.0", data_address, mpqa_version=2)
    # Gather the JSON file from MPQA.
    mpqa_json = obj.produce_json_file()
    data = obj.doc2csds(mpqa_json, json_output=True)

    n = len(data['csds_objects'])
    progressbar = -1
    for k in range(n):
        item = data['csds_objects'][k]

        word_based_info = char_to_word(
            text=item['text'], head=item['head'], start=item['head_start'], end=item['head_end'], clean=clean
        )
        item.update(word_based_info)

        item['target'] = find_info(item['target_link'], data['target_objects'], clean, parent_w_text=item['w_text'], verbose=verbose)
        item['nested_source'] = find_info(item['nested_source_link'], data['agent_objects'], clean, parent_w_text=item['w_text'], verbose=verbose)
        item['attitude'] = find_info(item['attitude_link'], data['csds_objects'], clean, add_attitude_attributes=True, parent_w_text=item['w_text'], verbose=verbose, data_targets=data['target_objects'])

        data['csds_objects'][k] = item

        if activate_progressbar and progressbar < k//(n//100):
            progressbar = k//(n//100)
            print(f'{progressbar}% completed')

    with open(save_address, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)



tokenize_and_extract_info(
    data_address='../mpqa_dataprocessing/database.mpqa.cleaned.221201',
    save_address='MPQA2.0_v221205_cleaned.json',
    clean=True,
    verbose=True
)



# tokenize_and_extract_info(
#     data_address='../mpqa_dataprocessing/database.mpqa.cleaned',
#     save_address='MPQA2.0_v221205.json',
#     clean=False,
#     verbose=False
# )
