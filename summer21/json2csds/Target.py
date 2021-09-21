
class Target:
    target_id = ""
    span_start = -1
    span_end = -1
    
    def __init__( self, this_id, this_span_start, this_span_end ):
        self.target_id = this_id
        self.span_start = this_span_start
        self.span_end = this_span_end
    
    