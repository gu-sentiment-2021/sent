class Agent:
    """
    This class is used for agent annotation type.
    """
    id = ""  # ID of this agent.
    nested_source = []  # The nested-source is a list of agent IDs beginning with the writer and ending with the ID
    # for the immediate agent being referenced.
    agent_uncertain = ""  # Used when the annotator is uncertain whether the agent is the correct source of a private

    # state/speech event.

    def __init__(self, id="", nested_source=[], agent_uncertain=""):
        """
        The init method (constructor) for Agent class.
        """
        self.id = id
        self.nested_source = nested_source
        self.agent_uncertain = agent_uncertain


