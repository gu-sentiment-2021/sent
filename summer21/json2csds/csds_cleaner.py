import re
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

import json
from json2csds import JSON2CSDS

from nltk.tokenize.treebank import TreebankWordDetokenizer
detokenizer = TreebankWordDetokenizer()

def clean_item(self, txt):
    re_pattern = '[a-zA-Z0-9 _=+/\"\'\-]'

    txt = re.sub('\n', '  ', txt)
    txt = re.sub('<UH>', 'UUHH', txt)
    txt = re.sub('<' + re_pattern + '*>', '', txt)
    txt = re.sub(re_pattern + '+>', '', txt)
    txt = re.sub('--', ' -- ', txt)

    return txt


def back_to_clean(lst):
    txt = detokenizer.detokenize(lst)
    txt = re.sub(' \.', '.', txt)
    return txt


def char_to_word(self, docname="", text="", head="", start=0, end=0, clean=False):
    if text.find(head) >= 0:
        text1 = text[0: start]
        text2 = text[start: end]
        text3 = text[end:]

        if clean:
            text_tokens1 = word_tokenize(self.__clean_item(text1))
            text_tokens2 = word_tokenize(self.__clean_item(text2))
            text_tokens3 = word_tokenize(self.__clean_item(text3))
            all_text_tokens = word_tokenize(self.__clean_item(text))
        else:
            text_tokens1 = word_tokenize(text1)
            text_tokens2 = word_tokenize(text2)
            text_tokens3 = word_tokenize(text3)
            all_text_tokens = word_tokenize(text)

        if all_text_tokens != text_tokens1 + text_tokens2 + text_tokens3:
            print(f"<Warning word tokenization mismatch in {docname}: {text_tokens2} | {all_text_tokens}/>")

        cleaned_text = back_to_clean(all_text_tokens)
        cleaned_head = back_to_clean(text_tokens2)

        # returns start index, list of tokens and the length of the tokens after the first index which should be considered
        return len(text_tokens1), len(text_tokens1) + len(text_tokens2), all_text_tokens, len(text_tokens2), cleaned_text, cleaned_head

clean_address = "..\mpqa_dataprocessing\database.mpqa.cleaned.221201"
clean_obj = JSON2CSDS("MPQA2.0", clean_address, mpqa_version=2)
# Gather the JSON file from MPQA.
clean_mpqa_json = clean_obj.produce_json_file()
clean_json_output = clean_obj.doc2csds(clean_mpqa_json, json_output=True)

# Path is where you want to save the JSON file.
path = ''

with open(path + 'MPQA.json', 'w', encoding='utf-8') as f:
    json.dump(clean_mpqa_json, f, ensure_ascii=False, indent=4)


with open(path + 'MPQA2.0_v221205_org.json', 'w', encoding='utf-8') as f:
    json.dump(clean_json_output, f, ensure_ascii=False, indent=4)

# Loading the saved JSON file.
with open(path + 'MPQA2.0_v221205_org.json', encoding='utf-8') as clean_json_file:
    clean_data = json.load(clean_json_file)

address = "..\mpqa_dataprocessing\database.mpqa.cleaned"
obj = JSON2CSDS("MPQA2.0", address, mpqa_version=2)
# Gather the JSON file from MPQA.
mpqa_json = obj.produce_json_file()
json_output = obj.doc2csds(mpqa_json, json_output=True)

# Path is where you want to save the JSON file.
path = ''

with open(path + 'MPQA.json', 'w', encoding='utf-8') as f:
    json.dump(mpqa_json, f, ensure_ascii=False, indent=4)


with open(path + 'MPQA2.0_v221205_org.json', 'w', encoding='utf-8') as f:
    json.dump(json_output, f, ensure_ascii=False, indent=4)

# Loading the saved JSON file.
with open(path + 'MPQA2.0_v221205_org.json', encoding='utf-8') as json_file:
    data = json.load(json_file)
