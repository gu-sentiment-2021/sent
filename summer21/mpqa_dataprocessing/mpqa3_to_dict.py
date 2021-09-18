# mpqa3_to_dict helps to convert MPQA stand-off format to python dictionaries.
# It provides the following functionalities:
# 1) Convert an MPQA document to a dictionary
# 2) Convert an entire corpus to a dictionary

class mpqa3_to_dict:
    """
    mpqa3_to_dict helps to convert MPQA stand-off format to python dictionaries.
    """

    corpus_name = "" # Name of the corpus from which the documents were drawn.
    mpqa_dir = "database.mpqa.3.0" # mpqa 3.0 root directory

    def __init__(self, corpus_name="", mpqa_dir="database.mpqa.3.0"):
        self.corpus_name = corpus_name
        self.mpqa_dir = mpqa_dir

    def doc_to_dict(self, doc_name):
        """
        It converts an MPQA document to a python dictionary.
        :param doc_name: The name of the document to be converted.
        :return: A python dictionary representing the document.
        """
        None

    def corpus_to_dict(self, doc_list):
        """
        It converts an entire list of MPQA documents to a python dictionary.
        :param doc_list: The list of document names to be converted.
        :return: A python dictionary representing the corpus.
        """
        None