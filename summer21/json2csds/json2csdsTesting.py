from sent.summer21.mpqa_dataprocessing.mpqa3_to_dict import mpqa3_to_dict
import json
from extended_csds import ExtendedCSDS, ExtendedCSDSCollection
from Target import sTarget, eTarget, Target


class JSON2CSDS:
    """
    This is a class in order to convert json file to csds.
    """

    def __init__(self, corpus_name="", mpqa_dir="database.mpqa.3.0"):
        """
        The init method (constructor) for JSON2CSDS class.
        """
        self.corpus_name = corpus_name
        self.mpqa_dir = mpqa_dir

    def produce_json_file(self):
        """
        It uses the mpqa3_to_dict module in order to convert mpqa to json (dict) file.
        :return: A json file which is obtained from mpqa corpus.
        """
        m2d = mpqa3_to_dict(self.corpus_name, self.mpqa_dir)
        result = m2d.corpus_to_dict()
        return result

    def type_mapper(self, key):
        """
        A mapping between types and their corresponding number based on the enum.
        :param key: type of the annotation.
        :return: Type's corresponding number based on the enum.
        """
        type_map_dict = {'sentiment': 1,
                         'arguing': 2,
                         'agreement': 3,
                         'intention': 4,
                         'speculation': 5,
                         'other-attitude': 6,
                         'expressive_subjectivity': 7,
                         'unknown': 8}
        return type_map_dict[key]

    def alert_wrong_annot(self, annot, doc_id, error=None):
        """
        It used for alert wrong annotation(s).
        :param annot: The annotation that error(s) were happening in its process.
        :param error: The error(s) that happened.
        """
        print('===================\nWrong annotation!!')
        print(annot)
        print('Error details: (doc_id: ' , doc_id, ')')
        print(error)
        print('===================')

    def go_get_targets(self, annots, target_id):
        """
        It goes after the targets in the target-links.
        :param annots: A Python dict which represents all annotations in the dic.
        :param target_id: Id of the target.
        :return: A python list of the targets.
        """
        targets = annots[target_id]['newETarget-link'] + annots[target_id]['sTarget-link']

        return targets

    def process_es(self, all_annot, es_annot, doc_id):
        """
        It processes an ES (expressive-subjectivity) type annotation.
        :param all_annot: A Python dict which represents all annotations in the dic.
        :param es_annot: A Python dict which represents an ES annotation.
        :param doc_id: Id of the doc.
        :return: A csds object.
        """
        # The following lines of code are under development!
        try:
            its_polarity = (es_annot['polarity'].split('-')[1] if es_annot['polarity'].find('-') >= 0 else es_annot[
                'polarity']) if 'polarity' in es_annot else es_annot['ese-type'].split('-')[1]

            csds_object = ExtendedCSDS(es_annot['sentence'],
                                       es_annot['span-in-sentence'][0],
                                       es_annot['span-in-sentence'][1],
                                       None,  # Belief!!!
                                       its_polarity,
                                       es_annot['intensity'],
                                       self.type_mapper('expressive_subjectivity'),
                                       this_head=self.go_get_targets(all_annot, es_annot['targetFrame-link']),
                                       this_doc_id=doc_id,
                                       this_sentence_id=es_annot['sentence-id']
                                       )

            arr_p = ['neutral', 'positive', 'negative', 'both', 'uncertain']
            if its_polarity not in arr_p:
                print('polarity: ', its_polarity)
            arr_i = ['low', 'medium', 'high', 'extreme']
            if es_annot['intensity'] not in arr_i:
                print('intensity: ', es_annot['intensity'])

        except Exception as err:
            self.alert_wrong_annot(es_annot, doc_id, error=err)
            return None
        return csds_object

    def doc2csds(self, json_file):
        """
        It converts a document annotation from json to csds
        :param json_file: The json file which is obtained form mpqa to json conversion.
        :return: A csds collection (several csds objects).
        """
        # List of all document names extracted from the json file
        doc_list = json_file['doclist']

        docs = json_file['docs']

        # In here, we create a csds collection that stores the csds objects
        ext_csds_coll = ExtendedCSDSCollection(self.corpus_name)

        # Process each document
        for doc_name in doc_list:
            curr_doc = docs[doc_name]

            # Extracts the list of all annotations
            annotations = curr_doc['annotations']

            # Check for "expressive-subjectivity" annotation type
            if 'expressive-subjectivity' in curr_doc:
                # In the following line of code, we extract the IDs of ES type annotations
                es_list = curr_doc['expressive-subjectivity']
                # Process each ES item by its corresponding ID
                for es_id in es_list:
                    annotation_item = annotations[es_id]
                    csds_object = self.process_es(annotations, annotation_item, doc_name)
                    # Store the object!
                    ext_csds_coll.add_labeled_instance(csds_object)
                    del csds_object
                del es_list

        return ext_csds_coll


########################
# test
address = "E:\Thesis\mpqa_3_0_database\database.mpqa.3.0"
obj = JSON2CSDS("MPQA3.0", address)
mpqa_json = obj.produce_json_file()
csds_coll_result = obj.doc2csds(mpqa_json)
