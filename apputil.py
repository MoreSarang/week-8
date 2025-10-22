from collections import defaultdict
import numpy as np


class MarkovText:
    """
    A simple Markov text generator.
    Builds a transition dictionary of words (states)
    and generates new text based on the Markov property.
    """

    def __init__(self, corpus: str):
        """
        Initialize with a text corpus.
        :param corpus: A string of text
        """
        self.corpus = corpus.split()  # tokenize by whitespace
        self.term_dict = None

    def get_term_dict(self):
        """
        Builds a transition dictionary where each unique token maps
        to a list of all words that follow it in the corpus.
        Example: {'is': ['very', 'she'], ...}
        """
        term_dict = defaultdict(list)

        for i in range(len(self.corpus) - 1):
            current_word = self.corpus[i]
            next_word = self.corpus[i + 1]
            term_dict[current_word].append(next_word)

        self.term_dict = dict(term_dict)

        # Keeping duplicates preserves the real transition frequency,
        # making next-word selection naturally probabilistic.

        return self.term_dict

    def generate(self, seed_term: str = None, term_count: int = 15) -> str:
        """
        Generates text using the Markov property.
        :param seed_term: Optional word to start the generation.
        :param term_count: Number of terms to generate.
        :return: A generated string of words.
        """
        if not self.term_dict:
            raise ValueError("Term dictionary is not built. Run get_term_dict() first.")

        if seed_term is None:
            seed_term = np.random.choice(list(self.term_dict.keys()))
        elif seed_term not in self.term_dict:
            raise ValueError(f"'{seed_term}' not found in corpus.")

        current_word = seed_term
        output = [current_word]

        for _ in range(term_count - 1):
            next_words = self.term_dict.get(current_word)
            if not next_words:
                # if no next word (end of chain), restart randomly
                current_word = np.random.choice(list(self.term_dict.keys()))
            else:
                current_word = np.random.choice(next_words)
            output.append(current_word)

        return " ".join(output)
