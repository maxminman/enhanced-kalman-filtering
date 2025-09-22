"""Stub implementation of RTS Smoother"""

class RTSSmoother:
    def __init__(self, config):
        self.config = config
    
    def smooth(self, states, covariances):
        return states, covariances