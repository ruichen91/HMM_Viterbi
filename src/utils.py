import numpy as np
from collections import Counter


def read_predict(pred_file):
    with open(pred_file) as f:
        tokens = f.read().splitlines()

    return tokens


def read_files(feature_label_file):
    transit_A = {}
    emission_B = {}

    keyFile = open(feature_label_file, 'r')
    key = keyFile.readlines()

    for i in range(len(key)):
        # identify each sentence ending
        key[i] = key[i].rstrip('\n').strip()
        if key[i] == "":
            continue

        # each line pasing
        keyFields = key[i].split('\t')
        if len(keyFields) != 2:
            print("format error in key at line " + str(i) + ":" + key[i])
            exit()

        # get word_POS correspondence
        keyToken = keyFields[0]
        keyPos = keyFields[1]
        update_dict(emission_B, keyPos, keyToken)

        # get POS_POS correspondence
        keyPosNext = ""
        key[i + 1] = key[i + 1].rstrip('\n').strip()
        if key[i + 1] != "" and key[i + 1] is not None:
            keyFieldsNext = key[i + 1].split('\t')
            keyPosNext = keyFieldsNext[1]
        update_dict(transit_A, keyPos, keyPosNext)

    # count and get probability
    prob_update(transit_A)
    prob_update(emission_B)

    return transit_A, emission_B


def prob_update(_dict_):
    for key, value in _dict_.items():
        cnt = Counter()
        siz = len(value)
        for word in value:
            cnt[word] += 1./siz
        _dict_[key] = dict(cnt)


def update_dict(_dict_, key, value):
    if key in _dict_:
        _dict_[key].append(value)
    else:
        _dict_[key] = [value]
