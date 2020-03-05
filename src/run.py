import numpy as np
from utils import *
from bigram_hmm import *
from trigram_hmm import *

def main():
    # training data
    filename = '../data/POS_dev_train.pos'
    # transit, emission = read_files_trigram(filename)
    transit, emission = read_files(filename)
    # validation data
    valid_file = '../data/POS_dev.words'
    valid_tokens = read_predict(valid_file)
    # test data
    test_file = '../data/POS_test.words'
    test_tokens = read_predict(test_file)

    # HMM model
    hmm = Bigram_HMM(transit, emission)
    # hmm = Trigram_HMM(transit, emission)
    hmm.predict(test_tokens, '../data/POS_test.pos')


if __name__ == '__main__':
    main()