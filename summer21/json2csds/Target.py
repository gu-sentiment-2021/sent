class Target:
    """
    This class is a parent class for eTarget(s) and sTarget(s).
    This class has target_id, span_start and span_end that both eTarget(s) and sTarget(s)
    include these parameters.
    """
    target_id = ""  # id of eTarget or sTarget
    span_start = -1  # start span of eTarget or sTarget
    span_end = -1  # end span of eTarget or sTarget

    def __init__(self, this_id, this_span_start, this_span_end):
        """
        The init method (constructor) for Target class.
        """
        self.target_id = this_id
        self.span_start = this_span_start
        self.span_end = this_span_end


class eTarget(Target):
    """
    This class is for entity/event level target(s).
    """
    type_etarget = ""  # type of eTarget, possible values: entity, event, other
    is_negated = False  # this is True when eTarget is nagated
    is_referred_in_span = False  # this is optional attribute

    def __init__(self, this_id, this_span_start, this_span_end, this_type_etarget,
                 this_is_negated=False, this_is_referred_in_span=False):
        """
        The init method (constructor) for eTarget class.
        """
        Target.__init__(self, this_id, this_span_start, this_span_end)
        self.type_etarget = this_type_etarget
        self.is_negated = this_is_negated
        self.is_referred_in_span = this_is_referred_in_span


class sTarget(Target):
    """
    This class is for span based target(s).
    """
    target_uncertain = ""  # Used when an annotator is uncertain about whether this is the correct target for the attitude.
    etarget_link = ""  # consist id of eTarget(s) that coverer by this sTarget span

    def __init__(self, this_id, this_span_start, this_span_end, this_target_uncertain="",
                 this_etarget_link=[]):
        """
        The init method (constructor) for sTarget class.
        """
        Target.__init__(self, this_id, this_span_start, this_span_end)
        self.target_uncertain = this_target_uncertain
        self.etarget_link = this_etarget_link


class TargetCollection:
    """
    Holds a collection of Target objects for a single corpus, each of which represents
    a single Target annotation in the corpus.
    """
    # List of target objects which are made from target annotations from the mpqa corpus.
    target_instances = []

    # A python dict that is used for indexing the target objects based on their ID
    indexer = dict()

    # Name of the corpus from which the objects in this collection were drawn.
    # This collection represents a single corpus.
    corpus = ""

    def __init__(self, this_corpus):
        """
        Stores the name of the corpus from which the objects are drawn.
        :param this_corpus:
        """
        self.corpus = this_corpus

    def add_instance(self, new_instance):
        """
        Adds a single Target object to the collection for an actual annotation.
        :param new_instance: The Target object representing the Target annotation.
        :return: None.
        """
        # Add the new Target instance to the list
        self.target_instances.append(new_instance)

        # Index the new Target instance based on its ID
        self.indexer[new_instance.target_id] = len(self.target_instances)

    def get_instance(self, instance_id):
        """
        Returns a single Target object from the collection by its ID, if not exists, None will be returned.
        :param instance_id: The ID of the Target annotation which is desired to be received.
        :return: None.
        """
        # Check if the instance_id is available in the collection by using the indexer in O(1) average time complexity
        if instance_id in self.indexer:
            # Return the corresponding Target object by using its ID.
            return self.target_instances[self.indexer[instance_id]]
        return None

    # Under construction: The following method is not using the storage efficiently! But it is not being used very much.
    def del_instance(self, instance_id):
        """
        Deletes a single Target object from the collection by its ID.
        :param instance_id: The ID of the Target annotation which is desired to be deleted.
        :return: None.
        """
        # Check if the instance_id is available in the collection by using the indexer in O(1) average time complexity
        if instance_id in self.indexer:
            # Put None in the corresponding place of the object which is going to be deleted.
            self.target_instances[self.indexer[instance_id]] = None

            # Delete the index value of the object which is going to be deleted.
            del self.indexer[instance_id]

    def reset_collection(self):
        """
        Reset (set empty) the Target objects collection.
        :return: None.
        """
        self.indexer = dict()
        self.target_instances = []
