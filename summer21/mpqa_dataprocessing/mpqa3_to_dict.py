# mpqa3_to_dict helps to convert MPQA stand-off format to python dictionaries.
# It provides the following functionalities:
# 1) Convert an MPQA document to a dictionary
# 2) Convert an entire corpus to a dictionary

import os
import re

HAS_LIST_OF_IDS = [ # These attributes may have any number of ids. (>= 0)
    "nested-source", "attitude-link", "insubstantial",
    "sTarget-link", "newETarget-link", "eTarget-link"
]

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
        # example: ./database.mpqa.3.0/docs/20011024/21.53.09-11428
        doc_lines = []
        with open(os.path.join(self.mpqa_dir, "docs", doc_name)) as doc_file:
            doc_lines = doc_file.readlines()
        
        # example: ./database.mpqa.3.0/man_anns/20011024/21.53.09-11428/gateman.mpqa.lre.3.0
        anno_lines = []
        with open(os.path.join(self.mpqa_dir, "man_anns", doc_name,
                  "gateman.mpqa.lre.3.0")) as anno_file:
            anno_lines = anno_file.readlines()

        # Final output
        output = {
            "agent": [],
            "expressive-subjectivity": [],
            "direct-subjective": [],
            "objective-speech-event": [],
            "attitude": [],
            "targetFrame": [],
            "sTarget": [],
            "eTarget": [],
            "sentence": [],
            "supplementaryAttitude": [],
            "supplementaryExpressive-subjectivity": [],
            "annotations": {}
        }

        # Process all annotation lines
        for anno in anno_lines:
            if len(anno) < 1: # If the line is empty then skip it.
                continue
            if anno[0] == '#': # If it is a comment then skip it.
                continue
            # Parsing the main components of an annotation line.
            line_id, span, anno_type, attributes = anno.split('\t')
            # Converting span to a tuple of ints.
            span = span.split(',')
            span = (int(span[0]), int(span[1]))
            # Removes ' \n' at the end of the string.
            attributes = attributes[:-2]
            # A temporary variable for an annotation line before knowing its ID.
            temp_dict = {
                'line-id': int(line_id),
                'span': span,
                'anno-type': anno_type,
            }
            # Process all attributes
            if len(attributes) == 0: # example: split annotation
                continue
            # Splits with the whitespaces out of the quotes as the delimeter
            attributes = re.split(r' (?=([^"]*"[^"]*")*[^"]*$)', attributes)
            for attribute in attributes:
                key, val = attribute.split('=')
                val = val[1:-1] # Removes double quotation marks
                if key in HAS_LIST_OF_IDS:
                    temp_dict[key] = [] if val == "none" else val.split(',')
                else:
                    temp_dict[key] = val
            # We probably know the identifier assigned to the annotation by now
            # except some of the agnets and the sentences
            id = temp_dict.get("id", line_id)
            temp_dict.pop("id", None)
            # Updating the final output
            output[id] = temp_dict
            if anno_type in output:
                output[anno_type].append(id)
            else: # If it's a new type of annotation, warn us in red!
                output[anno_type] = [id]
                print("\033[91m <UNKNOWN ANNO: {}>\033[00m" .format(anno_type))
        
        return output

    def corpus_to_dict(self, doc_list):
        """
        It converts an entire list of MPQA documents to a python dictionary.
        :param doc_list: The list of document names to be converted.
        :return: A python dictionary representing the corpus.
        """
        None