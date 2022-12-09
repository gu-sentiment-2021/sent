import re
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

import json
from json2csds import JSON2CSDS

from nltk.tokenize.treebank import TreebankWordDetokenizer

detokenizer = TreebankWordDetokenizer()



cache_clean_tokenizations_dict = {}
cache_tokenizations_dict = {}


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


def white_in_warning(text):
    return f'\033[00m{text}\033[93m'


def white_in_error(text):
    return f'\033[00m{text}\033[91m'



def clean_item(txt):
    re_pattern = '[a-zA-Z0-9 _=+/\"\'\-]'

    txt = re.sub('\n', '  ', txt)
    txt = re.sub('<UH>', 'UUHH', txt)
    txt = re.sub('<' + re_pattern + '*>', '', txt)
    txt = re.sub(re_pattern + '+>', '', txt)
    txt = re.sub('--', ' -- ', txt)
    txt = re.sub('  ', ' ', txt)

    # Handle ``s and ''s
    start_quotation_marks_indices = []
    end_quotation_marks_indices = []
    for i in range(len(txt) - 1):
        if txt[i:i + 2] == '``':  # or ord(txt[i]) == 39:
            start_quotation_marks_indices.append(i)
        if txt[i:i + 2] == "''":  # or ord(txt[i]) == 39:
            end_quotation_marks_indices.append(i)
    for i in range(len(start_quotation_marks_indices)):
        txt = txt[0: start_quotation_marks_indices[i]] + '"' + txt[start_quotation_marks_indices[i] + 2:]
    for i in range(len(end_quotation_marks_indices)):
        txt = txt[0: end_quotation_marks_indices[i]] + '"' + txt[end_quotation_marks_indices[i] + 2:]

    return txt


def back_to_clean(lst):
    txt = detokenizer.detokenize(lst)
    txt = re.sub(' \.', '.', txt)
    return txt


def cache_clean_tokenizations(text):
    if text not in cache_clean_tokenizations_dict:
        text2 = text + ' .'
        tokens = word_tokenize(clean_item(text2))
        tokens = list(map(clean_item, tokens))[0: -1]
        cache_clean_tokenizations_dict[text] = tokens
    return cache_clean_tokenizations_dict[text]


def cache_tokenizations(text):
    if text not in cache_clean_tokenizations_dict:
        text2 = text + ' .'
        tokens = word_tokenize(text2)[0:-1]
        cache_tokenizations_dict[text] = tokens
    return cache_tokenizations_dict[text]


def char_to_word(item_id="", text="", head="", start=0, end=0, clean=False, verbose=False):

    text1 = text[0: start]
    text2 = text[start: end]
    text3 = text[end:]

    if clean:
        text_tokens1 = cache_clean_tokenizations(text1)
        text_tokens2 = cache_clean_tokenizations(text2)
        # text_tokens3 = cache_clean_tokenizations(text3)
        all_text_tokens = cache_clean_tokenizations(text)
    else:
        text_tokens1 = cache_tokenizations(text1)
        text_tokens2 = cache_tokenizations(text2)
        # text_tokens3 = cache_tokenizations(text3)
        all_text_tokens = cache_tokenizations(text)

    if verbose and all_text_tokens[len(text_tokens1): len(text_tokens1) + len(text_tokens2)] != text_tokens2:
        print(
            f"\033[93m <Warning word tokenization mismatch id=<{white_in_warning(item_id)}>: \n\t head=<{white_in_warning(repr(text2))}> \n\t text=<{white_in_warning(repr(text))}> \n\t w_head={white_in_warning(text_tokens2)} \n\t w_text={white_in_warning(all_text_tokens)} \n /> \033[00m")

    # returns start index, list of tokens and the length of the tokens after the first index which should be considered
    return {
        'w_head_span': (len(text_tokens1), len(text_tokens1) + len(text_tokens2)),
        'w_text': all_text_tokens,
        'w_head': text_tokens2,
        'clean_text': back_to_clean(all_text_tokens),
        'clean_head': back_to_clean(text_tokens2)
    }


def find_info(ids, data_subset, clean=False, add_attitude_attributes=False, parent_w_text=[], parent_id='',
              verbose=False, data_targets={}):
    word_based_info = {}
    word_based_info_list = []
    if ids is None:
        return word_based_info_list
    for item_id in ids:
        if type(data_subset) is dict:
            if item_id in data_subset:
                item = data_subset[
                    item_id]  # dictionary: char based for sentence, word_based for sentence array, aspect, polarity, intensity, type
                word_based_info = char_to_word(
                    item_id=item_id, text=item['text'], head=item['head'], start=item['head_start'],
                    end=item['head_end'], clean=clean, verbose=verbose
                )
                if add_attitude_attributes:
                    word_based_info.update({
                        'annotation_type': item['annotation_type'],
                        'polarity': item['polarity'],
                        'intensity': item['intensity'],
                        'target': find_info(item['target_link'], data_targets, clean,
                                            parent_w_text=word_based_info['w_text'], parent_id=item_id, verbose=False)
                    })
                if verbose and parent_w_text != [] and word_based_info['w_text'] != [] and parent_w_text != \
                        word_based_info['w_text']:
                    print(
                        f'\033[91m <Error sentence mismatch parent_id=<{white_in_error(parent_id)}> & child_id=<{white_in_error(item_id)}>: \n\t parent_w_text=\t{white_in_error(parent_w_text)} \n\t child_w_text=\t{white_in_error(word_based_info["w_text"])} \n /> \033[00m')
            elif verbose:
                print(f"\033[93m <Warning id=<{white_in_warning(item_id)}> couldn't be found./> \033[00m\033[00m")
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
                            'target': find_info(item['target_link'], data_targets, clean,
                                                parent_w_text=word_based_info['w_text'], parent_id=item_id,
                                                verbose=False)
                        })
                    if verbose and parent_w_text != [] and word_based_info['w_text'] != [] and parent_w_text != \
                            word_based_info['w_text']:
                        print(
                            f'\033[91m <Error sentence mismatch parent_id=<{white_in_error(parent_id)}> & child_id=<{white_in_error(item_id)}>: \n\t parent_w_text=\t{white_in_error(parent_w_text)} \n\t child_w_text=\t{white_in_error(word_based_info["w_text"])} \n /> \033[00m')

        word_based_info_list.append(word_based_info)

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

        item_id = item['unique_id']
        item['target'] = find_info(item['target_link'], data['target_objects'], clean, parent_w_text=item['w_text'],
                                   parent_id=item_id, verbose=verbose)
        item['nested_source'] = find_info(item['nested_source_link'], data['agent_objects'], clean,
                                          parent_w_text=item['w_text'], parent_id=item_id, verbose=verbose)
        item['attitude'] = find_info(item['attitude_link'], data['csds_objects'], clean, add_attitude_attributes=True,
                                     parent_w_text=item['w_text'], parent_id=item_id, verbose=verbose,
                                     data_targets=data['target_objects'])

        data['csds_objects'][k] = item

        if activate_progressbar and progressbar < k // (n // 100):
            progressbar = k // (n // 100)
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
