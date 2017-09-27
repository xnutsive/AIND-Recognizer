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

    def __init__(self, word_indices, probs, language_model):
        self.word_indices = word_indices
        self.language_model = language_model

        # weight to use for the language model
        # predictions
        self.lm_alpha = 10

        # Seed minimal assumed log_s for sentences
        self.min_log_s = [0 for w in self.word_indices]
        self.min_log_s.append(0)

        self.keep_words = 10

        # Only keep 20 most likely used words per word in the
        # sentence.
        # This should reduce search space significantly.
        self.word_probabilities = [dict(sorted(p.items(), key=lambda x: x[1])[-self.keep_words:]) for p in probs]

        Problem.__init__(self, (0, None, '', 0, ''))

    def actions(self, state):
        actions = []
        next_word_index = self.word_indices[state[0]]

        for word, prob in self.word_probabilities[next_word_index].items():
            action = (state[0]+1, next_word_index, word, prob,
                ' '.join(state[4].split(' ')[-3:]) + ' ' + word)
            actions.append(action)

        return actions


    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""

        # Since actions and states use the same format to store
        # data, we can just return the action as a resulting state.
        return action


    def goal_test(self, state):
        return state[0] == len(self.word_indices)

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""

        # c is the cost of previous path segments
        # state2 is the latest word state. It doesn't have information on
        # the previous states or words.

        # print("considering " + state2[4])

        log_p = state2[3]
        log_s_lang = self.min_log_s[state2[0]]

        phrase = state2[4]
        if state2[0] == len(self.word_indices):
            phrase += " </s>"

        try:
            log_s_lang = self.language_model.log_s(phrase)
            if self.min_log_s[state2[0]] > log_s_lang:
                self.min_log_s[state2[0]] = log_s_lang
        except:
            pass

        return -( log_p + self.lm_alpha * log_s_lang )

    def value(self, state):
        """For optimization problems, each state has a value.  Hill-climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError
