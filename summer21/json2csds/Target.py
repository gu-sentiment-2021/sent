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
