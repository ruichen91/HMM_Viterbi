import numpy as np
from utils import read_files, read_predict
from HMM import *

def main():
    # training data
    filename = './POS_train.pos'
    transit, emission = read_files(filename)
    # validation data
    valid_file = './POS_dev.words'
    valid_tokens = read_predict(valid_file)
    # test data
    test_file = './POS_test.words'
    test_tokens = read_predict(test_file)

    # HMM model
    hmm = HMM(transit, emission)
    hmm.predict(valid_tokens, './POS_test.pos')
    a = 0

if __name__ == '__main__':
    main()