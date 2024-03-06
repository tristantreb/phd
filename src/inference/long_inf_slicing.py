from functools import reduce

import numpy as np
import pandas as pd


class SharedNodeVariable:
    def __init__(self, name, card, graph_key):
        self.name = name
        self.card = card
        self.graph_key = graph_key
        self.messages = {}
        self.posteriors = []

    def add_message(self, plate_key, message):
        assert message.shape == (
            self.card,
        ), "The message must have the same shape as the variable's cardinality"
        # Always replace the message, even if it already exists
        self.messages[plate_key] = message

    def get_virtual_message(self, plate_key):
        # Remove message with plate_key from the list of messages
        messages_up = self.messages.copy()
        if plate_key in self.messages.keys():
            messages_up.pop(plate_key)
        if len(messages_up) == 0:
            return None
        elif len(messages_up) == 1:
            return list(messages_up.values())[0]
        else:
            return reduce(np.multiply, list(messages_up.values()))
