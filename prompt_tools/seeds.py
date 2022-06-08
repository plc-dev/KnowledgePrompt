from nltk.corpus import wordnet as wn
from typing import List
import random as rng

def extract_leaf_hyponyms(synset: object) -> List[List]:
    """
    BFS for instance-hyponym extraction
    """
    visited_hyponyms = set()
    hyponym_stack = {hyponym for hyponym in synset.hyponyms()}

    instance_hypernyms = set()
    while(len(hyponym_stack)):
        current_hyponym = hyponym_stack.pop()
        current_hyponym_hyponyms = current_hyponym.hyponyms()

        if len(current_hyponym_hyponyms):
            for hyponym in current_hyponym_hyponyms:
                if hyponym not in visited_hyponyms:
                    hyponym_stack.add(hyponym)
        else:
            hypernyms = [hypernym.name for hypernym in current_hyponym.hypernyms()]
            for hypernym in hypernyms:
                if hypernym in instance_hypernyms:
                    instance_hypernyms.append(current_hyponym)
                else:
                    instance_hypernyms = [current_hyponym]

        visited_hyponyms.add(current_hyponym)

    return instance_hypernyms
