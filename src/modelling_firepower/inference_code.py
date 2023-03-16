import utils
import copy


class beliefPropagation:
    @staticmethod
    def observe(obs, key_value_pair):
        node_message = copy.copy(key_value_pair)
        for key, val in node_message.items():
            if key != obs: node_message[key] = 0
            else: node_message[key] = 1

        print("Observation:", node_message)
        return node_message

    @staticmethod
    def message_down(dist):
        """
        dist: marginal distribution
        """
        print("Message down: ", dist)
        return dist

    @staticmethod
    def factor_message_up(incoming_message, cpt_B_A):   
        """
        The factor mode message is the product of all received messages on the other edges, multiplied by the conditional probability distribution for the factor and summed over all variables except the one being sent to
        :param incoming_message: stochastic vector from A
        :param cpt_B_A:
        :return:
        """
        # TODO: add a function to invert a conditional probability table to merge factor_message_up and factor_message_down
        # compute sum_A( P(A|B) )
        factor_message = utils.get_second_level_keys(cpt_B_A)
        for key_A, items_B in cpt_B_A.items():
            for key_B, p_B_A in items_B.items():
                factor_message[key_B] += p_B_A * incoming_message[key_A]

        message = utils.normalise(factor_message)
        print("Message up:", message)

        return message

    @staticmethod
    def factor_message_down(incoming_message, cpt_B_A):
        """
        The factor mode message is the product of all received messages on the other edges, multiplied by the conditional probability distribution for the factor and summed over all variables except the one being sent to.
        :param incoming_message:
        :param cpt_B_A:
        :return:
        """
        # compute sum_B( P(A|B) )
        factor_message = utils.get_first_level_keys(cpt_B_A)
        for key_A, items_B in cpt_B_A.items():
            for key_B, p_B_A in items_B.items():
                factor_message[key_A] += p_B_A * incoming_message[key_B]

        message = utils.normalise(factor_message)
        print("Message down:", message)

        return message

    @staticmethod
    def node_message(messages):
        """
        The variable node message is the product of all received messages on the other edges
        :param messages: incoming variable node messages
        :return: variable node belief message
        """
        # All messages have the same keys
        belief = utils.get_first_level_keys(messages[0], init_value=1)

        for key in belief.keys():
            for message in messages:
                belief[key] *= message[key]

        # Normalise
        p = utils.normalise(belief)

        print('Node belief:', p)
        return p