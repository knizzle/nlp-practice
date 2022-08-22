from msilib import sequence
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'I love my dog',
    'i love my cat', # case insensitive
    'You love my dog!' # tokenizer strips punctuation
    'Do you think my dog is amazing?'
]

tokenizer = Tokenizer(num_words = 100, oov_token="<OOV") # oov = outer vocabulary
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

# print(word_index) 
# {'love': 1, 'my': 2, 'i': 3, 'dog': 4, 'cat': 5, 'you': 6}

sequences = tokenizer.texts_to_sequences(sentences)

# print(word_index) 
# {'my': 1, 'love': 2, 'dog': 3, 'i': 4, 'you': 5, 'cat': 6, 'do': 7, 'think': 8, 'is': 9, 'amazing': 10}

# print(sequences) 
# [[4, 2, 1, 3], [4, 2, 1, 6], [5, 2, 1, 3, 7, 5, 8, 1, 3, 9, 10]]

test_data = [
    'i really love my dog',
    'my dog loves my manatee'
]

test_seq = tokenizer.texts_to_sequences(test_data)
# print(test_seq)
# w/out oov >> [[4, 2, 1, 3], [1, 3, 1]] = 'i love my dog', 'my dog my' 
# w/oov >> [[5, 1, 3, 2, 4], [2, 4, 1, 2, 1]]