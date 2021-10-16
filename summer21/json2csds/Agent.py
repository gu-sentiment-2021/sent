class Agent:
    """
    This class is used for agent annotation type.
    """
    id = ""  # ID of this agent.
    nested_source = []  # The nested-source is a list of agent IDs beginning with the writer and ending with the ID.
    # for the immediate agent being referenced.
    agent_uncertain = ""  # Used when the annotator is uncertain whether the agent is the correct source of a private.
    doc_id = ""  # ID of the document.
    sentence_id = -1  # index of sentence within document.
    text = ""  # sentence in which the annotated head occurs.
    head_start = -1  # start span of agent.
    head_end = -1  # end span of agent.
    head = ""  # target of annotation.

    def __init__(self, id="", agent_uncertain="", text="", head_start=-1, head_end=-1,
                 head="", doc_id="", sentence_id=-1, nested_source=[]):
        """
        The init method (constructor) for Agent class.
        """
        self.id = id
        self.agent_uncertain = agent_uncertain
        self.doc_id = doc_id
        self.sentence_id = sentence_id
        self.text = text
        self.head_start = head_start
        self.head_end = head_end
        self.head = head
        self.nested_source = nested_source


class AgentCollection:
    """
    Holds a collection of Agent objects for a single corpus, each of which represents
    a single Agent annotation in the corpus.
    """
    # List of agent objects which are made from agent annotations from the MPQA corpus.
    agent_instances = []

    # A python dict that is used for indexing the agent objects based on their ID
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
        Adds a single Agent object to the collection for an actual annotation.
        :param new_instance: The Agent object representing the Agent annotation.
        :return: None.
        """
        # Add the new Agent instance to the list
        self.agent_instances.append(new_instance)

        # Index the new Agent instance based on its ID
        self.indexer[new_instance.id] = len(self.agent_instances)

    def get_instance(self, instance_id):
        """
        Returns a single Agent object from the collection by its ID, if not exists, None will be returned.
        :param instance_id: The ID of the Agent annotation which is desired to be received.
        :return: None.
        """
        # Check if the instance_id is available in the collection by using the indexer in O(1) average time complexity.
        if instance_id in self.indexer:
            # Return the corresponding Agent object by using its ID.
            return self.agent_instances[self.indexer[instance_id]]
        return None

    def get_all_instances(self):
        """
        Returns the list of Agent objects.
        :return: A list of Agent objects.
        """
        return self.agent_instances

    def get_num_instances(self):
        """
        Gets the number of Agent objects in this collection corresponding
        to actual annotations in the corpus.
        :return:  An integer, the count of Agent objects.
        """
        return len(self.agent_instances)

    # Under construction: The following method is not using the storage efficiently! But it is not being used very much.
    def del_instance(self, instance_id):
        """
        Deletes a single Agent object from the collection by its ID.
        :param instance_id: The ID of the Agent annotation which is desired to be deleted.
        :return: None.
        """
        # Check if the instance_id is available in the collection by using the indexer in O(1) average time complexity.
        if instance_id in self.indexer:
            # Put None in the corresponding place of the object which is going to be deleted.
            self.agent_instances[self.indexer[instance_id]] = None

            # Delete the index value of the object which is going to be deleted.
            del self.indexer[instance_id]

    def reset_collection(self):
        """
        Reset (set empty) the Agent objects collection.
        :return: None.
        """
        self.indexer = dict()
        self.agent_instances = []
