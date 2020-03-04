import numpy as np


class HMM:
    '''
        dictionary based HMM, that is,
        both prob matrix are actually dictionaries
    '''

    def __init__(self, transit_A, emission_B):
        # dictionary
        self.transit = transit_A
        self.emission = emission_B

        # states
        self.states = list(transit_A.keys())
        self.T = len(transit_A)

        # tokens
        self.num_tokens = 0
        self.smooth_coef = 1

    def prob_emiss_transit(self, condition, observe, emiss_or_transit):
        if emiss_or_transit == 'transition':
            sub_dict = self.transit[condition]
        else:
            sub_dict = self.emission[condition]

        if observe in sub_dict:
            return sub_dict[observe]
        else:
            return 1e-8

    def predict_each_sentence(self, sentence):
        ''' predict POS for each sentence '''
        path = np.zeros((self.T, len(sentence)))

        prev_path = np.ones(self.T)
        for w, word in enumerate(sentence):
            for c, curr in enumerate(self.states):
                path[c, w] = np.max([prev_path[p] *
                                    self.prob_emiss_transit(curr, word, 'emission') *
                                    self.prob_emiss_transit(prev, curr, 'transition')
                                    for p, prev in enumerate(self.states)])


            prev_path = path[:, w]

        # fetch POS for sentence
        idx = [np.argmax(path[:, column]) for column in range(len(sentence))]
        pos_tagging = [self.states[i] for i in idx]
        return pos_tagging

    def predict(self, tokens, save_path):
        ''' Viterbi algorithm
            sentence by sentence for prediction
        '''
        # get the number of all tokens
        for word in tokens:
            if word != '':
                self.num_tokens += 1


        POS_all_test = []
        Token_all_test = []
        sent = []
        POS_each_sent = []
        for it, token in enumerate(tokens):
            if token == '':
                pos = self.predict_each_sentence(sent)
                Token_all_test.append(sent)
                POS_all_test.append(pos)
                sent = []
                continue
            else:
                sent.append(token)

        self.save_prediction(Token_all_test, POS_all_test, save_path)

    def save_prediction(self, tokens, pos, filename):
        assert len(tokens) == len(pos) , 'file length should be the same'

        with open(filename, 'w') as f:
            for i in range(len(tokens)):
                tok_sent = tokens[i]
                pos_sent = pos[i]

                assert len(tok_sent) == len(pos_sent), 'sentence length should be the same'
                for j in range(len(tok_sent)):
                    f.write(tok_sent[j] + '\t' + pos_sent[j] + '\n')

                f.write('\n')




