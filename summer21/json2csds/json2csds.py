from sentiment.sent.summer21.mpqa_dataprocessing.mpqa3_to_dict import mpqa3_to_dict
import json
from extended_csds import ExtendedCSDS, CSDSCollection


class JSON2CSDS:
    """
    This is a class in order to convert json file to csds.
    """

    # A mapping between types and their corresponding number based on the enum
    type_mapper: {'sentiment': 1,
                  'arguing': 2,
                  'agreement': 3,
                  'intention': 4,
                  'speculation': 5,
                  'other_attitude': 6,
                  'expressive_subjectivity': 7}

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

    def process_agent(self, agent_annot):
        """
        It processes an agent type annotation.
        :param json_file: The json file which is obtained form mpqa to json conversion.
        :return: A csds object.
        """
        pass
        return None

    def process_es(self, es_annot, doc_id):
        """
        It processes an ES (expressive-subjectivity) type annotation.
        :param json_file: A Python dict which represents an ES annotation.
        :return: A csds object.
        """
        # The following lines of code are under development!

        csds_object = ExtendedCSDS(es_annot['sentence'],
                                   es_annot['span-in-sentence'][0],
                                   es_annot['span-in-sentence'][1],
                                   es_annot['polarity'],
                                   es_annot['intensity'],
                                   self.type_mapper['expressive_subjectivity'],
                                   this_head=es_annot['targetFrame-link'],
                                   this_doc_id=doc_id,
                                   this_sentence_id=es_annot['sentence-id']
                                   )
        return csds_object

    def process_ds(self, ds_annot, doc_id):
        """
        It processes a DS type annotation.
        :param json_file: A Python dict which represents a DS annotation.
        :return: A csds object.
        """

        return None

    # This method is not being used till now!
    def process_ose(self, ose_annot, doc_id):
        """
        It processes an OSE type annotation.
        :param json_file: A Python dict which represents an OSE annotation.
        :return: A csds object.
        """
        # !
        return None

    def process_att(self, att_annot, doc_id):
        """
        It processes an attitude type annotation.
        :param json_file: A Python dict which represents an attitude annotation.
        :return: A csds object.
        """
        # !
        return None

    def process_tf(self, tf_annot, doc_id):
        """
        It processes a target frame type annotation.
        :param json_file: A Python dict which represents a DS annotation.
        :return: A csds object.
        """
        # !
        return None

    def process_starget(self, starget_annot, doc_id):
        """
        It processes a sTarget type annotation.
        :param json_file: A Python dict which represents a sTarget annotation.
        :return: A csds object.
        """
        # !
        return None

    def process_etarget(self, etarget_annot, doc_id):
        """
        It processes an eTarget type annotation.
        :param json_file: A Python dict which represents an eTarget annotation.
        :return: A csds object.
        """
        # !
        return None

    def process_sentence(self, sentence_annot, doc_id):
        """
        It processes a sentence type annotation.
        :param json_file: A Python dict which represents a sentence annotation.
        :return: A csds object.
        """
        # !
        return None

    def doc2csds(self, json_file):
        """
        It converts a document annotation from json to csds
        :param json_file: The json file which is obtained form mpqa to json conversion.
        :return: A csds collection (several csds objects).
        """
        # List of all document names extracted from the json file
        doc_list = json_file['doclist']

        docs = json_file['docs']

        # In here, there must be some code to create a csds collection that stores the csds objects
        # -------------

        # Process each document
        for doc_name in doc_list:
            curr_doc = docs[doc_name]

            # Extracts the list of all annotations
            annotations = curr_doc['annotations']

            # Check for "agent" annotation type
            if 'agent' in curr_doc:
                # In the following line of code, we extract the IDs of agent type annotations
                agent_list = curr_doc['agent']
                # Process each agent item by its corresponding ID
                for agent_id in agent_list:
                    annotation_item = agent_list[agent_id]
                    csds_object = self.process_agent(annotation_item)
                    # must store the object!
                    del annotation_item
                del agent_list
            # Check for "expressive-subjectivity" annotation type
            if 'expressive-subjectivity' in curr_doc:
                # In the following line of code, we extract the IDs of ES type annotations
                es_list = curr_doc['expressive-subjectivity']
                # Process each ES item by its corresponding ID
                for es_id in es_list:
                    annotation_item = es_list[es_id]
                    csds_object = self.process_es(annotation_item, doc_name)
                    # must store the object!
                    del csds_object
                del es_list
            # Check for "direct-subjective" annotation type
            if 'direct-subjective' in curr_doc:
                # In the following line of code, we extract the IDs of DS type annotations
                ds_list = curr_doc['direct-subjective']
                # Process each DS item by its corresponding ID
                for ds_id in ds_list:
                    annotation_item = ds_list[ds_id]
                    csds_object = self.process_ds(annotation_item, doc_name)
                    # must store the object!
                    del csds_object
                del ds_list
            # Check for "objective-speech-event" annotation type, this part is not activated yet!]
            '''
            if 'objective-speech-event' in curr_doc:
                # In the following line of code, we extract the IDs of OSE type annotations
                ose_list = curr_doc['objective-speech-event']
                # Process each OSE item by its corresponding ID
                for ose_id in ose_list:
                    annotation_item = ose_list[ose_id]
                    csds_object = self.process_ose(annotation_item)
                    # must store the object!
                    del csds_object
                del ose_list
            '''
            # Check for "attitude" annotation type
            if 'attitude' in curr_doc:
                # In the following line of code, we extract the IDs of attitude type annotations
                att_list = curr_doc['attitude']
                # Process each attitude item by its corresponding ID
                for att_id in att_list:
                    annotation_item = att_list[att_id]
                    csds_object = self.process_att(annotation_item)
                    # must store the object!
                    del csds_object
                del att_list
            # Check for "targetFrame" annotation type
            if 'targetFrame' in curr_doc:
                # In the following line of code, we extract the IDs of target frame type annotations
                tf_list = curr_doc['targetFrame']
                # Process each target frame item by its corresponding ID
                for tf_id in tf_list:
                    annotation_item = tf_list[tf_id]
                    csds_object = self.process_tf(annotation_item)
                    # must store the object!
                    del csds_object
                del tf_list
            # Check for "sTarget" annotation type
            if 'sTarget' in curr_doc:
                # In the following line of code, we extract the IDs of sTarget type annotations
                starget_list = curr_doc['sTarget']
                # Process each sTarget item by its corresponding ID
                for starget_id in starget_list:
                    annotation_item = starget_list[starget_id]
                    csds_object = self.process_starget(annotation_item)
                    # must store the object!
                    del csds_object
                del starget_list
            # Check for "eTarget" annotation type
            if 'eTarget' in curr_doc:
                # In the following line of code, we extract the IDs of eTarget type annotations
                etarget_list = curr_doc['eTarget']
                # Process each eTarget item by its corresponding ID
                for etarget_id in etarget_list:
                    annotation_item = etarget_list[etarget_id]
                    csds_object = self.process_etarget(annotation_item)
                    # must store the object!
                    del csds_object
                del etarget_list
            # Check for "sentence" annotation type
            if 'sentence' in curr_doc:
                # In the following line of code, we extract the IDs of sentence type annotations
                sentence_list = curr_doc['sentence']
                # Process each sentence item by its corresponding ID
                for sentence_id in sentence_list:
                    annotation_item = sentence_list[sentence_id]
                    csds_object = self.process_sentence(annotation_item)
                    # must store the object!
                    del csds_object
                del sentence_list

            del annotations
        return None


########################
# test
address = "F:\pych\mpqa_3_0_database\database.mpqa.3.0"
obj = JSON2CSDS("MPQA3.0", address)
jsf = obj.produce_json_file()
