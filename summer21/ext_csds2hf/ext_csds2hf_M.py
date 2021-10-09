from sent.summer21.json2csds import json2csds
import transformers
from transformers import AutoTokenizer


class extCsds2hfV1:
    """
        This class is for using csds in hugging face models!
    """

    def __init__(self, corpus_name="", mpqa_dir="database.mpqa.3.0"):
        """
        The init method (constructor) for extCsds2hfV1 class.
        """
        self.corpus_name = corpus_name
        self.mpqa_dir = mpqa_dir

    def get_mpqa_results(self):
        """
        It uses the mpqa_dir attribute in order to get the results of mpqa->json->csds conversion.
        :return: A pair of collections:
        1. A csds collection (several csds objects).
        2. A target collection (several target objects).
        """
        json2csds_obj = json2csds.JSON2CSDS(corpus_name=self.corpus_name, mpqa_dir=self.mpqa_dir)
        mpqa_json_file = json2csds_obj.produce_json_file()
        csds_objs, target_objs = json2csds_obj.doc2csds(mpqa_json_file)
        return csds_objs, target_objs

    def extract_csds_collection(self, tuple_result):
        """
        It receives a tuple which is returned by mpqa->csds conversion procedure.
        :return: A collection of csds objects.
        """
        csds_coll = tuple_result[0]
        return csds_coll

    def extract_csds_objects(self, tuple_result):
        """
        It receives a csds collection and returns a list of csds objects.
        :return: A collection of csds objects.
        """
        csds_coll = self.extract_csds_collection(tuple_result)
        csds_objs = csds_coll.get_all_instances()
        return csds_objs[0]

def findHead(sequence, head_text, symbols):
    head_start = -1
    head_end = -1
    j = 0

    for symbol in symbols:
        sequence = sequence.replace(symbol, '')

    tokenized_sequence = sequence.split()
    print('sequence: ', tokenized_sequence)

    tokenized_head_text = head_text.split()
    print('head_text: ', tokenized_head_text)

    for i in range(len(tokenized_sequence)):
        if j <= len(tokenized_head_text):
            if tokenized_head_text[j] == tokenized_sequence[i]:
                if j == 0:
                    head_start = i
                if j == len(tokenized_head_text) - 1:
                    head_end = i
                j += 1
            else:
                j = 0
                head_start = -1
        else:
            break

    return head_start, head_end

def findHeadInToken(tokens, head_start, head_end, symbols):
    head_start_in_model = -1
    head_end_in_model = -1
    j = 0
    for i in range(len(tokens)):
        flag = True
        if j == head_start:
            head_start_in_model = i
        if j == head_end:
            head_end_in_model = i
        if type(tokens[i]) is int:
            if tokens[i] == 101 or tokens[i] == 102: #need?
                j = j
        elif tokens[i].startswith('##'):
            j = j
        else:
            for symbol in symbols:
                if tokens[i] == symbol:
                    flag = False
            if flag == True:
                j += 1
    return head_start_in_model, head_end_in_model

def tokenizer_and_model(sequence, head_text):
    checkpoint = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    tokens = tokenizer.tokenize(sequence)
    print('tokens (output of tokenize): ', tokens)
    tokens_model = tokenizer.prepare_for_model(tokens)
    print('tokens_model (output of prepare tokens for model): ', tokens_model['input_ids'])

    # for highlight head in tokens
    symbols = ['.', '!', ',', '?'] #extend?
    # find head in sequence
    head_start, head_end = findHead(sequence, head_text, symbols)
    print('head_start: ', head_start, ', head_end: ', head_end)

    # find pos of head in tokens
    head_start_in_model, head_end_in_model = findHeadInToken(tokens_model['input_ids'], head_start, head_end, symbols)
    print('head_start_in_model: ', head_start_in_model, ', head_end_in_model: ', head_end_in_model)
    print('check head_start/end_in_model: ')
    for i in range(head_start_in_model, head_end_in_model + 1):
        print(tokens_model['input_ids'][i])

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    print('input_ids (output of convert tokens to ids): ', input_ids)
    final_inputs = tokenizer.prepare_for_model(input_ids)
    print('final_inputs (output of prepare input_ids for model): ', final_inputs)

    return head_start_in_model, head_end_in_model, final_inputs

# test part
address = "..\mpqa_dataprocessing\databases\database.mpqa.3.0.cleaned"
obj = extCsds2hfV1("MPQA3.0", address)
tuple_res = obj.get_mpqa_results()

csds_objects = obj.extract_csds_objects(tuple_res)
print(csds_objects[0])

# tokenizer part
sequence = "This is first sentence. Using a Transformer network is simple!"
head_text = "is simple"
head_start_in_model, head_end_in_model, final_inputs = tokenizer_and_model(sequence, head_text)
