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


# test part
address = "..\mpqa_dataprocessing\databases\database.mpqa.3.0.cleaned"
obj = extCsds2hfV1("MPQA3.0", address)
tuple_res = obj.get_mpqa_results()

csds_objects = obj.extract_csds_objects(tuple_res)

print(csds_objects[0])

# for example
text = "Using a Transformer network is simple!"
head = "is simple"

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
raw_inputs = [text, head]
model_inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(model_inputs)
