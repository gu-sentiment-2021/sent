from sent.summer21.mpqa_dataprocessing.mpqa3_to_dict import mpqa3_to_dict
import json
from extended_csds import ExtendedCSDS, ExtendedCSDSCollection
from Target import sTarget, eTarget, Target, TargetCollection


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
        It is used for alerting wrong annotation(s).
        :param annot: The annotation that error(s) were happening in its process.
        :param error: The error(s) that happened.
        """
        print('===================\nWrong annotation!!')
        print(annot)
        print('Error details: (doc_id: ', doc_id, ')')
        print(error)
        print('===================')

    def go_get_targets(self, annots, target_id):
        """
        It goes after the targets in the target-links and fetches them.
        :param annots: A Python dict which represents all annotations in the doc.
        :param target_id: ID of the target.
        :return: A python list of the targets.
        """
        targets = annots[target_id]['newETarget-link'] + annots[target_id]['sTarget-link']

        return targets

    def process_agent(self, agent_annot, doc_id):
        """
        It processes an agent type annotation.
        :param agent_annot: A Python dict which represents an agent annotation.
        :param doc_id: ID of the doc.
        :return: A csds object.
        """
        try:
            # Extract features from agent annotation.
            if agent_annot['span-in-doc'][0] == 0 and agent_annot['span-in-doc'][1] == 0:
                its_text = None
                its_head_start = -1
                its_head_end = -1
                its_sentence_id = -1
            else:
                its_text = agent_annot['text']
                its_head_start = agent_annot['span-in-sentence'][0]
                its_head_end = agent_annot['span-in-sentence'][1]
                its_sentence_id = agent_annot['sentence-id']

            # Create a csds object based on the values of the agent annotation.
            csds_object = ExtendedCSDS(this_text=its_text,
                                       this_head_start=its_head_start,
                                       this_head_end=its_head_end,
                                       this_belief=None,
                                       this_polarity=None,
                                       this_intensity=None,
                                       this_annotation_type=self.type_mapper('unknown'),
                                       this_head=agent_annot['head'],
                                       this_doc_id=doc_id,
                                       this_sentence_id=its_sentence_id
                                       )
        except Exception as err:
            self.alert_wrong_annot(agent_annot, doc_id, error=err)
            return None
        return csds_object

    def process_es(self, all_annot, es_annot, doc_id):
        """
        It processes an ES (expressive-subjectivity) type annotation.
        :param all_annot: A Python dict which represents all annotations in the doc.
        :param es_annot: A Python dict which represents an ES annotation.
        :param doc_id: ID of the doc.
        :return: A csds object.
        """

        # Extract features from ES annotation.
        try:
            its_polarity = (es_annot['polarity'].split('-')[1] if es_annot['polarity'].find('-') >= 0 else es_annot[
                'polarity']) if 'polarity' in es_annot else es_annot['ese-type'].split('-')[1]

            # Because this is an optional attribute in ES.
            if 'intensity' in es_annot:
                its_intensity = es_annot['intensity']
            else:
                its_intensity = None

            # Create a csds object based on the values of the ES annotation.
            csds_object = ExtendedCSDS(this_text=es_annot['text'],
                                       this_head_start=es_annot['span-in-sentence'][0],
                                       this_head_end=es_annot['span-in-sentence'][1],
                                       this_belief=None,
                                       this_polarity=its_polarity,
                                       this_intensity=its_intensity,
                                       this_annotation_type=self.type_mapper('expressive_subjectivity'),
                                       this_target_link=self.go_get_targets(all_annot, es_annot['targetFrame-link']),
                                       this_head=es_annot['head'],
                                       this_doc_id=doc_id,
                                       this_sentence_id=es_annot['sentence-id']
                                       )
        except Exception as err:
            self.alert_wrong_annot(es_annot, doc_id, error=err)
            return None
        return csds_object

    def process_att(self, all_annot, att_annot, doc_id):
        """
        It processes an attitude type annotation.
        :param all_annot: A Python dict which represents all annotations in the doc.
        :param att_annot: A Python dict which represents an attitude annotation.
        :param doc_id: ID of the doc.
        :return: A csds object.
        """
        its_pol = None
        its_type = self.type_mapper('unknown')

        # Extract features from attitude annotation.
        try:
            if att_annot['attitude-type'].find('other') >= 0:
                its_type = self.type_mapper('other-attitude')
            else:
                # Extract polarity and type: check all of the cases such as corner cases.
                if att_annot['attitude-type'].find('-') != -1:
                    its_attitude_type = att_annot['attitude-type'].split('-')
                    length = len(its_attitude_type)
                    its_pol = 'positive' if its_attitude_type[length - 1].find('pos') >= 0 else 'negative'
                    its_type = self.type_mapper('agreement') if its_attitude_type[length - 2].find(
                        'agree') >= 0 else self.type_mapper(its_attitude_type[length - 2])
                else:
                    its_type = self.type_mapper(att_annot['attitude-type'])

            # Extract features from attitude annotation.
            csds_object = ExtendedCSDS(this_text=att_annot['text'],
                                       this_head_start=att_annot['span-in-sentence'][0],
                                       this_head_end=att_annot['span-in-sentence'][1],
                                       this_belief=None,
                                       this_polarity=its_pol,
                                       this_intensity=att_annot['intensity'],
                                       this_annotation_type=its_type,
                                       this_target_link=self.go_get_targets(all_annot, att_annot['targetFrame-link']),
                                       this_head=att_annot['head'],
                                       this_doc_id=doc_id,
                                       this_sentence_id=att_annot['sentence-id']
                                       )
            return csds_object

        except Exception as err:
            self.alert_wrong_annot(att_annot, doc_id, error=err)
            return None

    def process_tf(self, tf_annot, tf_id, doc_id):
        """
        It processes a target frame type annotation.
        :param tf_annot: A Python dict which represents a targetFrame annotation.
        :param tf_id: ID of the targetFrame.
        :return: A target (target frame) object.
        """
        try:
            # Extract features from target frame annotation and create an object from them.
            target_object = Target(this_id=tf_id,
                                   this_span_start=tf_annot['span-in-sentence'][0],
                                   this_span_end=tf_annot['span-in-sentence'][1])
            return target_object
        except Exception as err:
            self.alert_wrong_annot(tf_annot, doc_id, error=err)
            return None

    def process_starget(self, starget_annot, starget_id, doc_id):
        """
        It processes a sTarget type annotation.
        :param starget_annot: A Python dict which represents a sTarget annotation.
        :param starget_id: ID of the sTarget.
        :return: A sTarget object.
        """
        try:
            # Extract features from sTarget annotation and create an object from them.
            starget_object = sTarget(this_id=starget_id,
                                     this_span_start=starget_annot['span-in-sentence'][0],
                                     this_span_end=starget_annot['span-in-sentence'][1],
                                     this_etarget_link=starget_annot['eTarget-link'])

            # Check 'target-uncertain' which is an optional attribute.
            if 'target-uncertain' in starget_annot:
                starget_object.target_uncertain = starget_annot['target-uncertain']
        except Exception as err:
            self.alert_wrong_annot(starget_annot, doc_id, error=err)
            return None

        return starget_object

    def process_etarget(self, etarget_annot, etarget_id, doc_id):
        """
        It processes an eTarget type annotation.
        :param etarget_annot: A Python dict which represents an eTarget annotation.
        :param etarget_id: ID of the eTarget.
        :return: An eTarget object.
        """
        try:
            # Extract features from eTarget annotation and create an object from them.
            etarget_object = eTarget(this_id=etarget_id,
                                     this_span_start=etarget_annot['span-in-sentence'][0],
                                     this_span_end=etarget_annot['span-in-sentence'][1],
                                     this_type_etarget=etarget_annot['type'])

            # Check 'isNegated' and 'isReferredInSpan which are optional attributes.
            if 'isNegated' in etarget_annot:
                etarget_object.is_negated = etarget_annot['isNegated']

            if 'isReferredInSpan' in etarget_annot:
                etarget_object.is_referred_in_span = etarget_annot['isReferredInSpan']

        except Exception as err:
            self.alert_wrong_annot(etarget_annot, doc_id, error=err)
            return None

        return etarget_object

    '''
    def process_ds(self, ds_annot, doc_id):
        """
        It processes a DS (direct-subjective) type annotation.
        :param ds_annot: A Python dict which represents a DS annotation.
        :param doc_id: Id of the doc.
        :return: A csds object.
        """
        try:
            csds_object = ExtendedCSDS(this_text=ds_annot['text'],
                                       this_head_start=ds_annot['span-in-sentence'][0],
                                       this_head_end=ds_annot['span-in-sentence'][1],
                                       this_belief=None,
                                       # or maybe the polarity by getting it via attitude-link?
                                       this_polarity=ds_annot['attitude-type'],
                                       this_intensity=ds_annot['intensity'],
                                       this_annotation_type=self.type_mapper('expressive_subjectivity'),  # NOT SURE!
                                       this_target_link=ds_annot['attitude-link'],
                                       this_head=ds_annot['head'],
                                       this_doc_id=doc_id,
                                       this_sentence_id=ds_annot['sentence-id']
                                       )
            return csds_object
        except Exception as err:
            self.alert_wrong_annot(ds_annot, doc_id, error=err)
            return None

    # This method is not being used till now!
    def process_ose(self, ose_annot, doc_id):
        """
        It processes an OSE type annotation.
        :param ose_annot: A Python dict which represents an OSE annotation.
        :param doc_id: Id of the doc.
        :return: A csds object.
        """
        # !
        return None

    # This method is not being used till now!
    def process_sentence(self, sentence_annot, doc_id):
        """
        It processes a sentence type annotation.
        :param sentence_annot: A Python dict which represents a sentence annotation.
        :param doc_id: Id of the doc.
        :return: A csds object.
        """
        # !
        return None
    '''

    def doc2csds(self, json_file):
        """
        It converts a document annotation from json to csds and target.
        :param json_file: The json file which is obtained from mpqa to json conversion.
        :return: A pair of collections:
        1. A csds collection (several csds objects).
        2. A target collection (several target objects).
        """
        # List of all document names extracted from the json file.
        doc_list = json_file['doclist']

        docs = json_file['docs']

        # In here, we create a csds collection that stores the csds objects.
        ext_csds_coll = ExtendedCSDSCollection(self.corpus_name)
        # And here, we create a target collection that stores the target objects.
        target_coll = TargetCollection(self.corpus_name)

        # Process each document
        for doc_name in doc_list:
            curr_doc = docs[doc_name]

            # Extracts the list of all annotations.
            annotations = curr_doc['annotations']

            # Check for "agent" annotation type.
            if 'agent' in curr_doc:
                # In the following line of code, we extract the IDs of agent type annotations.
                agent_list = curr_doc['agent']
                # Process each agent item by its corresponding ID.
                for agent_id in agent_list:
                    annotation_item = annotations[agent_id]
                    csds_object = self.process_agent(annotation_item, doc_name)
                    # Store the object!
                    ext_csds_coll.add_labeled_instance(csds_object)
                    del annotation_item
                del agent_list
            # Check for "expressive-subjectivity" annotation type.
            if 'expressive-subjectivity' in curr_doc:
                # In the following line of code, we extract the IDs of ES type annotations.
                es_list = curr_doc['expressive-subjectivity']
                # Process each ES item by its corresponding ID.
                for es_id in es_list:
                    annotation_item = annotations[es_id]
                    csds_object = self.process_es(annotations, annotation_item, doc_name)
                    # Store the object!
                    ext_csds_coll.add_labeled_instance(csds_object)
                    del csds_object
                del es_list
            # Check for "attitude" annotation type.
            if 'attitude' in curr_doc:
                # In the following line of code, we extract the IDs of attitude type annotations.
                att_list = curr_doc['attitude']
                # Process each attitude item by its corresponding ID.
                for att_id in att_list:
                    annotation_item = annotations[att_id]
                    csds_object = self.process_att(annotations, annotation_item, doc_name)
                    ext_csds_coll.add_labeled_instance(csds_object)
                    del csds_object
                del att_list
            # Check for "targetFrame" annotation type.
            if 'targetFrame' in curr_doc:
                # In the following line of code, we extract the IDs of target frame type annotations.
                tf_list = curr_doc['targetFrame']
                # Process each target frame item by its corresponding ID.
                for tf_id in tf_list:
                    annotation_item = annotations[tf_id]
                    tf_object = self.process_tf(annotation_item, tf_id, doc_name)
                    # Store the object.
                    target_coll.add_instance(tf_object)
                    del tf_object
                del tf_list
            # Check for "sTarget" annotation type.
            if 'sTarget' in curr_doc:
                # In the following line of code, we extract the IDs of sTarget type annotations.
                starget_list = curr_doc['sTarget']
                # Process each sTarget item by its corresponding ID.
                for starget_id in starget_list:
                    annotation_item = annotations[starget_id]
                    starget_object = self.process_starget(annotation_item, starget_id, doc_name)
                    # Store the object.
                    target_coll.add_instance(starget_object)
                    del starget_object
                del starget_list
            # Check for "eTarget" annotation type.
            if 'eTarget' in curr_doc:
                # In the following line of code, we extract the IDs of eTarget type annotations.
                etarget_list = curr_doc['eTarget']
                # Process each eTarget item by its corresponding ID.
                for etarget_id in etarget_list:
                    annotation_item = annotations[etarget_id]
                    etarget_object = self.process_etarget(annotation_item, etarget_id, doc_name)
                    # Store the object.
                    target_coll.add_instance(etarget_object)
                    del etarget_object
                del etarget_list
            '''
            # Check for "direct-subjective" annotation type
            if 'direct-subjective' in curr_doc:
                # In the following line of code, we extract the IDs of DS type annotations
                ds_list = curr_doc['direct-subjective']
                # Process each DS item by its corresponding ID
                for ds_id in ds_list:
                    annotation_item = annotations[ds_id]
                    csds_object = self.process_ds(annotation_item, doc_name)
                    # Store the object!
                    ext_csds_coll.add_labeled_instance(csds_object)
                    del csds_object
                del ds_list
            # Check for "objective-speech-event" annotation type, this part is not activated yet!
            if 'objective-speech-event' in curr_doc:
                # In the following line of code, we extract the IDs of OSE type annotations
                ose_list = curr_doc['objective-speech-event']
                # Process each OSE item by its corresponding ID
                for ose_id in ose_list:
                    annotation_item = annotations[ose_id]
                    csds_object = self.process_ose(annotation_item)
                    # must store the object!
                    # WHAT TO DO?
                    del csds_object
                del ose_list
            # Check for "sentence" annotation type, this part is not activated yet!
            if 'sentence' in curr_doc:
                # In the following line of code, we extract the IDs of sentence type annotations
                sentence_list = curr_doc['sentence']
                # Process each sentence item by its corresponding ID
                for sentence_id in sentence_list:
                    annotation_item = annotations[sentence_id]
                    csds_object = self.process_sentence(annotation_item)
                    # must store the object!
                    # WHAT TO DO?
                    del csds_object
                del sentence_list

            del annotations
            '''
        return ext_csds_coll, target_coll


########################
# test
address = "..\mpqa_dataprocessing\databases\database.mpqa.3.0.cleaned"
obj = JSON2CSDS("MPQA3.0", address)
mpqa_json = obj.produce_json_file()
csds_coll_result, _ = obj.doc2csds(mpqa_json)
