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
    lines = keyFile.readlines()

    for i in range(len(lines)):
        # identify each sentence ending
        lines[i] = lines[i].rstrip('\n').strip()
        if lines[i] == "":
            continue

        # each lines pasing
        keyFields = lines[i].split('\t')
        if len(keyFields) != 2:
            print("format error in lines at lines " + str(i) + ":" + lines[i])
            exit()

        # get word_POS correspondence
        keyToken = keyFields[0]
        keyPos = keyFields[1]
        update_dict(emission_B, keyPos, keyToken)

        # get POS_POS correspondence
        keyPosNext = ""
        lines[i + 1] = lines[i + 1].rstrip('\n').strip()
        if lines[i + 1] != "" and lines[i + 1] is not None:
            keyFieldsNext = lines[i + 1].split('\t')
            keyPosNext = keyFieldsNext[1]
        update_dict(transit_A, keyPos, keyPosNext)

    # count and get probability
    prob_update(transit_A)
    prob_update(emission_B)

    return transit_A, emission_B


def read_files_trigram(feature_label_file):
    transit_A = {}
    emission_B = {}

    keyFile = open(feature_label_file, 'r')
    lines = keyFile.readlines()

    # file_size
    siz = len(lines)
    for i in range(siz):
        # identify each sentence ending
        line0 = lines[i].rstrip('\n').strip()
        if line0 == "":
            continue

        # each lines pasing
        keyFields = line0.split('\t')
        if len(keyFields) != 2:
            print("format error in lines at lines " + str(i) + ":" + lines[i])
            exit()

        # get word_POS correspondence
        keyToken = keyFields[0]
        key_current = keyFields[1]
        update_dict(emission_B, key_current, keyToken)

        # get POS_POS correspondence NEXT_1
        line1 = ""
        key_next = ""
        if i + 1 < siz - 1:
            line1 = lines[i + 1].rstrip('\n').strip()
        if line1 != "" and line1 is not None:
            content_next_1 = line1.split('\t')
            key_next = content_next_1[1]

        # get Pos_pos correspondence Next_2
        line2 = ""
        pos_value = ""
        if i + 2 < siz - 1:
            line2 = lines[i + 2].rstrip('\n').strip()
        if line2 != "" and line2 is not None:
            content_next_2 = line2.split('\t')
            pos_value = content_next_2[1]
        update_dict(transit_A, (key_current, key_next), pos_value)

    # count and get probability
    prob_update(transit_A)
    prob_update(emission_B)

    return transit_A, emission_B


def prob_update(_dict_):
    for lines, value in _dict_.items():
        cnt = Counter()
        siz = len(value)
        for word in value:
            cnt[word] += 1. / siz
        _dict_[lines] = dict(cnt)


def update_dict(_dict_, key, value):
    if key in _dict_:
        _dict_[key].append(value)
    else:
        _dict_[key] = [value]
