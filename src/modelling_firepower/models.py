# Lung Function Model
import utils


class lungModel:
    # Names
    """
    i: inflammation
    bl: bacterial load
    w: wellness
    """
    # Marginal distributions
    marginal_i = {
        "absent": 0.3,
        "small": 0.5,
        "heavy": 0.2
    }
    # Conditional probability table (cpt) of varA knowing varB
    cpt_bl_i = {
        "low": {
            "absent": 0.6,
            "small": 0.3,
            "heavy": 0.1
        },
        "medium": {
            "absent": 0.2,
            "small": 0.3,
            "heavy": 0.3
        },
        "high": {
            "absent": 0.2,
            "small": 0.4,
            "heavy": 0.6
        }
    }

    cpt_w_i = {
        1: {
            "absent": 0.01,
            "small": 0.2,
            "heavy": 0.4
        },
        2: {
            "absent": 0.09,
            "small": 0.3,
            "heavy": 0.3
        },
        3: {
            "absent": 0.2,
            "small": 0.3,
            "heavy": 0.15
        },
        4: {
            "absent": 0.3,
            "small": 0.1,
            "heavy": 0.1
        },
        5: {
            "absent": 0.4,
            "small": 0.1,
            "heavy": 0.05
        }
    }

    cpt_FEV1_bl = {
        "low": {
            'low': 0.6,
            'medium': 0.3,
            'high': 0.1,
        },
        "medium": {
            'low': 0.3,
            'medium': 0.4,
            'high': 0.4,
        },
        "high": {
            'low': 0.1,
            'medium': 0.3,
            'high': 0.5,
        }
    }

    def sample(self):
        inflammation = utils.threeStatesSample(list(self.marginal_i.values()),
                                               list(self.marginal_i.keys()))

        wellness = utils.nStatesSample(utils.get_stochastic_vector(self.cpt_w_i, inflammation),
                                       list(self.cpt_w_i.keys()))

        bacterial_load = utils.threeStatesSample(utils.get_stochastic_vector(self.cpt_bl_i, inflammation),
                                                 list(self.cpt_bl_i.keys()))

        FEV1 = utils.threeStatesSample(utils.get_stochastic_vector(self.cpt_FEV1_bl, bacterial_load),
                                       list(self.cpt_FEV1_bl.keys()))

        return {
            "Inflammation": inflammation,
            "Wellness": wellness,
            "Bacterial load": bacterial_load,
            "FEV1": FEV1
        }
