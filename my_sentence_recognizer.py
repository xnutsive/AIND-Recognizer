from asl_data import SinglesData
import arpa

from aimacode.search import Problem, Node


class SentenceProblem(Problem):

    """
    This class represents a list of ASL recognized word probabilities
    in a single sentence as a UCS path search problem.

    Main purpose of this approach is to improve words recognition by applying
    SLM joint word probabilities within one sentence.

    Problem graph is structured this way:
        1. Initial position is a start of a sentence, sometimes represented by
            symbol "<s>"
        2. Goal position is reached when the last word in the sentence reached.
        3. Each state is a word.
        4. Link cost is inverted log probability of word A + inverted log
            probability of word B + interted log joint probability of "A B".

    """

    def __init__(self, word_indices, word_probabilities, language_model):
        self.word_indices = word_indices
        self.word_probabilities = word_probabilities
        self.language_model = language_model

        Problem.__init__(self, tuple([0]))  # FIXME is empty state OK for init?

    def actions(self, state):
        actions = []
        next_word_index = self.word_indices[state[0]]

        for word, prob in self.word_probabilities[next_word_index].items():
            actions.append((state[0] + 1,next_word_index,word,prob))

        return actions


    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        return action

    def goal_test(self, state):
        return state[0] == len(self.word_indices)

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""

        # TODO actually use a language model here.
        # TODO apply sentence start and sentence end

        log_p = sum([s[3] for s in [state1, state2] if len(s) > 1])

        return 1/log_p
        # return c + 1

    def value(self, state):
        """For optimization problems, each state has a value.  Hill-climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError
