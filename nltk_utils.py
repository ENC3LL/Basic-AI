import numpy as np
import nltk
# nltk.download('punkt') # Uncomment this line if you get an error about 'punkt'

def tokenize(sentence):
    """
    Split sentence into array of words/tokens.
    a token can be a word or punctuation character, or number.
    """
    return nltk.word_tokenize(sentence)

def bag_of_words(tokenized_sentence, all_words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise.
    example:
    sentence = ["hello", "how", "are", "you"]
    words    = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog      = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word (optional, we use simple lower() here for simplicity)
    sentence_words = [word.lower() for word in tokenized_sentence]
    
    # initialize bag with 0 for each word
    bag = np.zeros(len(all_words), dtype=np.float32)
    
    for idx, w in enumerate(all_words):
        if w in sentence_words: 
            bag[idx] = 1.0
            
    return bag
