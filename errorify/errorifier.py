"""Synthetic data generator."""
import math
import pickle
import random
from numpy.random import choice as npchoice

VERBS = pickle.load(open('verbs.p', 'rb'))
COMMON_INSERTS = set(pickle.load(open('common_inserts.p', 'rb')))
COMMON_REPLACES = pickle.load(open('common_replaces.p', 'rb'))
COMMON_DELETES = pickle.load(open('common_deletes.p','rb'))

class Errorifier:
    """Generate errors in good sentences!"""

    def __init__(self, sentence: str):
        self.original_sentence = sentence.rstrip()
        self.sentence = self.original_sentence
        self.tokenized = None
        self.tokenize()

    def tokenize(self):
        self.tokenized = self.sentence.split()

    def correct(self):
        return self.original_sentence

    def no_error(self):
        return ' '.join(self.tokenized)

    def delete_error(self):
        if len(self.tokenized) > 0:
            insertable = list(range(len(self.tokenized)))
            index = random.choice(insertable)
            

            plist = list(COMMON_DELETES.values())
            plistsum = sum(plist)
            plist = [x / plistsum for x in plist]

            # Choose a bad word
            ins_word = npchoice(list(COMMON_DELETES.keys()), p=plist)
            self.tokenized.insert(index,ins_word)

        return ' '.join(self.tokenized)


    def verb_error(self, redir=True):
        """Introduce a verb error from morphs.txt."""

        if len(self.tokenized) > 0:
            verbs = [i for i, w in enumerate(self.tokenized) if w in VERBS]
            if not verbs:
                if redir:
                    return self.replace_error(redir=False)
                return self.sentence

            index = random.choice(verbs)
            word = self.tokenized[index]
            if not VERBS[word]:
                return self.sentence
            repl = random.choice(VERBS[word])
            self.tokenized[index] = repl

        return ' '.join(self.tokenized)

    def insert_error(self):
        """Delete a commonly inserted word."""
        if len(self.tokenized) > 1:
            deletable = [i for i, w in enumerate(self.tokenized) if w in COMMON_INSERTS]
            if not deletable:
                return self.sentence

            index = random.choice(deletable)
            del self.tokenized[index]
        return ' '.join(self.tokenized)

    def replace_error(self, redir=True):
        """Add a common replace error."""
        if len(self.tokenized) > 0:
            deletable = [i for i, w in enumerate(self.tokenized) if w in COMMON_REPLACES]
            if not deletable:
                if redir:
                    return self.verb_error(redir=False)
                return self.sentence

            index = random.choice(deletable)
            word = self.tokenized[index]
            if not COMMON_REPLACES[word]:
                return self.sentence

            # Normalize probabilities
            plist = list(COMMON_REPLACES[word].values())
            plistsum = sum(plist)
            plist = [x / plistsum for x in plist]

            # Choose a bad word
            repl = npchoice(list(COMMON_REPLACES[word].keys()), p=plist)
            self.tokenized[index] = repl

        return ' '.join(self.tokenized)

    def error(self):
        """Introduce a random error."""

        #count = math.floor(pow(random.randint(1, 11), 2) / 50) + 1
        count = npchoice([0,1,2,3,4],p=[0.05,0.07,0.25,0.35,0.28]) #original (a1)
        #count = npchoice([0,1,2,3,4],p=[0.1,0.1,0.2,0.3,0.3]) # (a2)
        #count = npchoice([0,1,2,3,4,5],p=[0.1,0.1,0.2,0.2,0.2,0.2]) # (a3)
        #count = npchoice([0,1,2,3,4,5],p=[0.1,0.1,0.2,0.2,0.2,0.2]) # (a4)
        #count = npchoice([0,1,2,3,4,5],p=[0.0,0.0,0.25,0.25,0.25,0.25]) # (a5)

        for x in range(count):
            # Note: verb_error redirects to replace_error and vice versa if nothing happened
            error_probs = [.30,.25,.25,.20] #original (a1)
            #error_probs = [.25,.30,.30,.15] # (a2)
            #error_probs = [.40,.25,.25,.10] #(a3)
            #error_probs = [.30,.30,.30,.10] #(a4)
            #error_probs = [.35,.25,.25,.15] #(a5)

            error_fun = npchoice([self.insert_error, self.verb_error, self.replace_error, self.delete_error],p=error_probs)
            self.sentence = error_fun()
            self.tokenize()

        return self.sentence
