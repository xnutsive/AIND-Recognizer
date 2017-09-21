import warnings
from asl_data import SinglesData
import arpa

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for item in range(test_set.num_items):
        probs = {}
        best_prob = float("-inf")
        best_guess = ""
        for model_name in models:
            try:
                score = models[model_name].score(*test_set.get_item_sequences(item))
                probs[model_name] = score
                if score > best_prob:
                    best_prob = score
                    best_guess = model_name

            except:
                probs[model_name] = float("-inf")

        probabilities.append(probs)
        guesses.append(best_guess)

    return (probabilities, guesses)
