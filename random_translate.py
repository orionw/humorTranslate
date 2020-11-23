import random
import nltk
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.corpus import words

from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import pandas as pd
import numpy as np

ALL_NOUNS = list(wn.all_synsets(wn.NOUN))
ALL_ADJ = list(wn.all_synsets(wn.ADJ))
ALL_VERBS = list(wn.all_synsets(wn.VERB))

def nltk_synset_to_word(synset: nltk.corpus.reader.wordnet.Synset) -> str:
    """ A helper function to turn a synset to the actual word """
    return synset.name().split(".")[0].replace("_", " ")

def random_translate(text: str, prob: float = 0.25) -> str:
    """
    An intelligent random baseline to randomly change verbs, nouns, or adjectives with `prob` probability.
    Will keep the same capitalization format as the original word.
    Uses NLTK to get random verbs, nouns, or adjectives

    Input:
        text: the string to translate randomly
        prob: the probability with which to flip words

    Returns:
        the randomly translated string
    """
    final_sent = text
    tokens = nltk.word_tokenize(text)
    tokens_tag = pos_tag(tokens)
    for token, pos in tokens_tag:
        if pos in ["NN", "NNP", "JJ", "VB", 'VBN', "VBG"]:
            if random.random() < prob:
                token_caps = token[0].isupper()
                if pos in ["NN", "NNP"]:
                    new_word = nltk_synset_to_word(random.choice(ALL_NOUNS))
                if pos == "JJ":
                    new_word = nltk_synset_to_word(random.choice(ALL_ADJ))
                if pos in ["VB", "VBN", "VBG"]:
                    new_word = nltk_synset_to_word(random.choice(ALL_VERBS))
                chosen_word = new_word if not token_caps else new_word[0].upper() + new_word[1:]
                final_sent = final_sent.replace(token, chosen_word, 1)
    return final_sent


if __name__ == "__main__":
    print(random_translate("Could Roy Moore Be Expelled From The Senate If Elected?"))
    print(random_translate("Connecticut pastor charged with stealing $8G in electricity"))


