from sent.summer21.json2csds import json2csds


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


# test part
address = "..\mpqa_dataprocessing\databases\database.mpqa.3.0.cleaned"
obj = extCsds2hfV1("MPQA3.0", address)
