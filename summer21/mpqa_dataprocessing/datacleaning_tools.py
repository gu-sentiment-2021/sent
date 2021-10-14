import os
import clipboard as cb
from mpqa3_to_dict import mpqa3_to_dict

def adjustSpansInRange(start_byte=0, end_byte=1e9, delta=0, filename="gateman.mpqa.lre.3.0"):
    """
    It adjusts the spans of the annotation lines in the 'filename' that are
    between 'start_byte' and 'end_byte', and adds 'delta' to them. It stores
    the modified file in the clipboard and prints the modified line ids.
    """
    modified_file = '' # Store the entire modified file
    ids = [] # Store the modified annotation line ids (to use them in the new README file)
    with open(os.path.join(filename)) as doc_file:
        doc_text = doc_file.readlines()
        for line in doc_text:
            if line[0] == '#': # Skip comment lines without modifing them
                modified_file += line
                continue
            id, span, anno_type, attr = line.split('\t')
            x, y = span.split(',')
            x, y = int(x), int(y)
            modified = False # Remember if we've modified this annotation line
            if x >= start_byte and x <= end_byte:
                x += delta
                modified = True
            if y >= start_byte and y <= end_byte:
                y += delta
                modified = True
            modified_file += '{}\t{},{}\t{}\t{}'.format(id, x, y, anno_type, attr)
            if modified: # Store its id, if we've modified this annotation line
                ids.append(int(id))
    cb.copy(modified_file) # Copy the complete modified file into the clipboard
    print("len(ids):", len(ids))
    print("ids:", ids)

# Sample:
# adjustSpansInRange(start_byte=2318, delta=+2) # You should run it inside the same directory of the annotation file.

def findCutPhrases(
    start_doc, end_doc=None, expand=False,
    mpqa_dir="mpqa_dataprocessing\\databases\\database.mpqa.3.0.cleaned", doclist_filename='doclist'
):
    """
    Find the phrases that their spans are potentially wrong.
    It can be one of these 2 types:
    1) There's whitespace at the first or the last character of the head.
    2) There's non-whitespace character immediately before the first character
    or immediately after the last character of the head.
    :param start_doc: First document id we're going to observe.
    :param end_doc: Last document id we're going to observe. start_doc is being
    used, if end_doc is not set.
    :param expand: prints the dictionary of each potentially broken annotation.
    :return: None
    """
    if end_doc is None:
        end_doc = start_doc
    m2d = mpqa3_to_dict(mpqa_dir=mpqa_dir)
    mpqadict = m2d.corpus_to_dict(doclist_filename=doclist_filename)
    alphabet = 'qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890'
    for doc in mpqadict['doclist'][start_doc:end_doc+1]: # Iterate over docs
        cut_phrases = []
        print(doc) # The name of the doc we're processing right now
        for v in mpqadict['docs'][doc]['annotations'].values(): # iterate over all annotations
            if 'implicit' in v.keys(): # skip implicit annotations
                if v['implicit'] == 'true':
                    continue
            if v['span-in-doc'][0] == v['span-in-doc'][1]: # skip annotations of zero length
                continue
            if 'text' in v.keys():
                # Check if there's non-whitespace character immediately before the first
                # character or immediately after the last character of the head.
                if v['span-in-sentence'][0] > 0 and v['text'][v['span-in-sentence'][0]-1] in alphabet\
                or v['span-in-sentence'][1] < len(v['text']) and v['text'][v['span-in-sentence'][1]] in alphabet:
                    cut_phrases += ['{}:{}'.format(v['line-id'], v['head'])]
                    if expand:
                        print(v)
                # Check if there's whitespace at the first or the last character of the head.
                if v['text'][v['span-in-sentence'][0]] not in alphabet+'`"(\'' \
                or v['text'][v['span-in-sentence'][1]-1] not in alphabet+'."%?\'),]':
                    cut_phrases += ['{}:{}'.format(v['line-id'], v['head'])]
                    if expand:
                        print(v)
        print(repr(cut_phrases))
        print()

# Sample:
# findCutPhrases(0, 69, expand=True)

def CountAllAttributeTypes(
    mpqa_dir="mpqa_dataprocessing\\databases\\database.mpqa.3.0.cleaned", doclist_filename='doclist'
):
    """
    It counts all types of attributs available in all annotation lines in all documents of a corpus.
    :return: a python dictionary with attribute names as keys and number of times they appeared in annotation files as values
    """
    m2d = mpqa3_to_dict(mpqa_dir=mpqa_dir)
    mpqadict = m2d.corpus_to_dict(doclist_filename=doclist_filename)
    attr_types = {}
    for doc in mpqadict['doclist']: # Iterate over all docs
        for anno_id, attributes in mpqadict['docs'][doc]['annotations'].items(): # Iterate over all annotations
            for attr_type, attr_value in attributes.items(): # Iterate over all attributes
                attr_types[attr_type] = attr_types.get(attr_type, 0) + 1 # Counter
    return attr_types

# Sample:
# print(CountAllAttributeTypes())