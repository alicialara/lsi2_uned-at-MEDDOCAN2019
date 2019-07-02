import io
import pickle
from os import listdir
from os.path import isfile, join
from shutil import copyfile

import numpy as np
from gensim.models import FastText, KeyedVectors

from aluned_nn.validation import compute_f1
from keras.models import Model, load_model
from keras.layers import TimeDistributed, Conv1D, Dense, Embedding, Input, Dropout, LSTM, Bidirectional, MaxPooling1D, \
    Flatten, concatenate
from aluned_nn.prepro import readfile, createBatches, createMatrices, iterate_minibatches, addCharInformation, padding
from keras.utils import plot_model
from keras.initializers import RandomUniform
from keras.optimizers import SGD, Nadam

"""Initialise class"""


class CNN_BLSTM(object):

    def __init__(self, EPOCHS, DROPOUT, DROPOUT_RECURRENT, LSTM_STATE_SIZE, CONV_SIZE, LEARNING_RATE, OPTIMIZER):

        self.epochs = EPOCHS
        self.dropout = DROPOUT
        self.dropout_recurrent = DROPOUT_RECURRENT
        self.lstm_state_size = LSTM_STATE_SIZE
        self.conv_size = CONV_SIZE
        self.learning_rate = LEARNING_RATE
        self.optimizer = OPTIMIZER

    def loadData(self):
        """Load data and add character information"""
        self.trainSentences = readfile("../main/neuralnetwork_input/train.txt")  # sentences list: words[word][BIO-TARGET]
        self.devSentences = readfile("../main/neuralnetwork_input/dev.txt")
        self.testSentences = readfile("../main/neuralnetwork_input/test.txt")
        # self.devSentences_lines = readfile("../main/neuralnetwork_input/dev_with_lines.txt")
        # self.testSentencesMain = readfile("../main/neuralnetwork_input/test_scenario_main.txt")
        # c_i = 0
        # for i in self.trainSentences:
        #     c_j = 0
        #     for j in i:
        #         if j[0] == "":
        #             self.trainSentences[c_i].pop(c_j)
        #             print(self.trainSentences[c_i])
        #         if j[1] == "":
        #             self.trainSentences[c_i][c_j][1] = "X"
        #         if j[2] == "":
        #             self.trainSentences[c_i][c_j][2] = "O"
        #         c_j += 1
        #     c_i += 1
        #
        # c_i = 0
        # for i in self.devSentences:
        #     c_j = 0
        #     for j in i:
        #         if j[0] == "":
        #             self.devSentences[c_i].pop(c_j)
        #             print(self.devSentences[c_i])
        #         if j[1] == "":
        #             self.devSentences[c_i][c_j][1] = "X"
        #         if j[2] == "":
        #             self.devSentences[c_i][c_j][2] = "O"
        #         c_j += 1
        #     c_i += 1
        #
        # c_i = 0
        # for i in self.testSentences:
        #     c_j = 0
        #     for j in i:
        #         if j[0] == "":
        #             self.testSentences[c_i].pop(c_j)
        #             print(self.testSentences[c_i])
        #         if j[1] == "":
        #             self.testSentences[c_i][c_j][1] = "X"
        #         if j[2] == "":
        #             self.testSentences[c_i][c_j][2] = "O"
        #         c_j += 1
        #     c_i += 1
        # self.trainSentences = readfile("data/train.txt")  # sentences list: words[word][BIO-TARGET]
        # self.devSentences = readfile("data/test.txt")
        # self.testSentences = readfile("data/trial.txt")

    def addCharInfo(self):
        # format: [['EU', ['E', 'U'], 'B-ORG\n'], ...]
        self.trainSentences = addCharInformation(self.trainSentences)
        self.devSentences = addCharInformation(self.devSentences)
        self.testSentences = addCharInformation(self.testSentences)
        # self.devSentences_lines = addCharInformation(self.devSentences_lines)
        # self.testSentencesMain = addCharInformation(self.testSentencesMain)

    def embed(self):
        """Create word- and character-level embeddings"""

        labelSet = set()
        postagSet = set()
        words = {}

        # unique words and labels in data (and pos tags...)
        # for dataset in [self.trainSentences, self.devSentences, self.testSentences, self.testSentencesMain]:
        for dataset in [self.trainSentences, self.devSentences, self.testSentences]:
            for sentence in dataset:
                for token, postag, char, label in sentence:
                    # token ... token, postag ... list of postags, char ... list of chars, label ... BIO labels
                    labelSet.add(label)
                    postagSet.add(postag)
                    words[token.lower()] = True

        # mapping for labels
        self.label2Idx = {}
        for label in labelSet:
            self.label2Idx[label] = len(self.label2Idx)

        self.postag2Idx = {}
        for postag in postagSet:
            if postag == '' or postag == '_':
                postag = 'X'
            self.postag2Idx[postag] = len(self.postag2Idx)

        self.postagEmbeddings = np.identity(len(self.postag2Idx), dtype='float32')  # identity matrix used

        # mapping for token cases
        case2Idx = {'numeric': 0, 'allLower': 1, 'allUpper': 2, 'initialUpper': 3, 'other': 4, 'mainly_numeric': 5,
                    'contains_digit': 6, 'PADDING_TOKEN': 7}
        self.caseEmbeddings = np.identity(len(case2Idx), dtype='float32')  # identity matrix used

        # read GLoVE word embeddings
        word2Idx = {}
        self.wordEmbeddings = []

        # _base_wikipedia2vec_300
        fEmbeddings = open("embeddings/wikipedia2vec/eswiki_20180420_300d.txt", encoding="utf-8")

        # loop through each word in embeddings
        count_lines = 0
        for line in fEmbeddings:
            if count_lines > 0:
                split = line.strip().split(" ")
                word = split[0]  # embedding word entry
                if len(word2Idx) == 0:  # add padding+unknown
                    word2Idx["PADDING_TOKEN"] = len(word2Idx)
                    vector = np.zeros(len(split) - 1)  # zero vector for 'PADDING' word
                    self.wordEmbeddings.append(vector)

                    word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
                    vector = np.random.uniform(-0.25, 0.25, len(split) - 1)
                    self.wordEmbeddings.append(vector)

                if split[0].lower() in words:
                    vector = np.array([float(num) for num in split[1:]])
                    self.wordEmbeddings.append(vector)  # word embedding vector
                    word2Idx[split[0]] = len(word2Idx)  # corresponding word dict
            count_lines += 1

        self.wordEmbeddings = np.array(self.wordEmbeddings)

        # dictionary of all possible characters
        self.char2Idx = {"PADDING": 0, "UNKNOWN": 1}
        for c in "Kqz~²EÉ<(ñ»©ÍNªe=vß!7W)Ø';ìμλ$*¥ft#Jδ³8\cíuD¡¿Hº≥†FQÓøÚÑüaZ’4?:●pÁë≈Â–3ièγ°hI[STá2 @bηÇκ·Υµ9y/dOα≤.òsÒ•,RX¼_-6k+wY′x%½U&∧óL{|éàŀ→1“”×Vr`ç5jä™Δã«^nMg}´>0m±χPAlιåoCB]®êâGúôöβ—\"":
            self.char2Idx[c] = len(self.char2Idx)

        # format: [[wordindices], [caseindices], [padded word indices], [label indices]]
        self.train_set = padding(createMatrices(self.trainSentences, word2Idx, self.label2Idx, case2Idx, self.char2Idx, self.postag2Idx))
        self.dev_set = padding(createMatrices(self.devSentences, word2Idx, self.label2Idx, case2Idx, self.char2Idx, self.postag2Idx))
        self.test_set = padding(createMatrices(self.testSentences, word2Idx, self.label2Idx, case2Idx, self.char2Idx, self.postag2Idx))
        # self.test_set_main = padding(createMatrices(self.testSentencesMain, word2Idx, self.label2Idx, case2Idx, self.char2Idx, self.postag2Idx))
        # self.test_set_main = []
        self.idx2Label = {v: k for k, v in self.label2Idx.items()}
        self.word2Idx_ = word2Idx
        self.case2Idx_ = case2Idx

        # onlyfiles = [f for f in listdir('../main/dev_complete.pkl') if isfile(join('../main/dev_complete.pkl', f))]
        # onlyfiles.sort()
        # data_w_file = {}
        # for file in onlyfiles:
        #     data_w_file[file] = padding(createMatrices(self.devSentences, word2Idx, self.label2Idx, case2Idx, self.char2Idx, self.postag2Idx))

        self.save_data_to_pickle()

    def createBatches(self):
        """Create batches"""
        self.train_batch, self.train_batch_len = createBatches(self.train_set)
        self.dev_batch, self.dev_batch_len = createBatches(self.dev_set)
        self.test_batch, self.test_batch_len = createBatches(self.test_set)

    def tag_dataset(self, dataset, model):
        """Tag data with numerical values"""
        correctLabels = []
        predLabels = []
        for i, data in enumerate(dataset):
            tokens, casing, char, labels, postags = data
            tokens = np.asarray([tokens])
            casing = np.asarray([casing])
            char = np.asarray([char])
            postag = np.asarray([postags])
            pred = model.predict([tokens, casing, char, postag], verbose=False)[0]
            pred = pred.argmax(axis=-1)  # Predict the classes
            correctLabels.append(labels)
            predLabels.append(pred)
        return predLabels, correctLabels

    def buildModel(self):
        """Model layers"""

        # character input
        character_input = Input(shape=(None, 52,), name="Character_input")
        embed_char_out = TimeDistributed(
            Embedding(len(self.char2Idx), 30, embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)),
            name="Character_embedding")(
            character_input)

        dropout = Dropout(self.dropout)(embed_char_out)

        # CNN
        conv1d_out = TimeDistributed(
            Conv1D(kernel_size=self.conv_size, filters=30, padding='same', activation='tanh', strides=1),
            name="Convolution")(dropout)
        maxpool_out = TimeDistributed(MaxPooling1D(52), name="Maxpool")(conv1d_out)
        char = TimeDistributed(Flatten(), name="Flatten")(maxpool_out)
        char = Dropout(self.dropout)(char)

        # word-level input
        words_input = Input(shape=(None,), dtype='int32', name='words_input')
        words = Embedding(input_dim=self.wordEmbeddings.shape[0], output_dim=self.wordEmbeddings.shape[1],
                          weights=[self.wordEmbeddings],
                          trainable=False)(words_input)

        # case-info input
        casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
        casing = Embedding(output_dim=self.caseEmbeddings.shape[1], input_dim=self.caseEmbeddings.shape[0],
                           weights=[self.caseEmbeddings],
                           trainable=False)(casing_input)

        # postag-info input
        postag_input = Input(shape=(None,), dtype='int32', name='postag_input')
        postagging = Embedding(output_dim=self.postagEmbeddings.shape[1], input_dim=self.postagEmbeddings.shape[0],
                               weights=[self.postagEmbeddings],
                               trainable=False)(postag_input)

        # concat & BLSTM
        output = concatenate([words, casing, char, postagging])
        output = Bidirectional(LSTM(self.lstm_state_size,
                                    return_sequences=True,
                                    dropout=self.dropout,  # on input to each LSTM block
                                    recurrent_dropout=self.dropout_recurrent  # on recurrent input signal
                                    ), name="BLSTM")(output)
        output = TimeDistributed(Dense(len(self.label2Idx), activation='softmax'), name="Softmax_layer")(output)

        # set up model
        self.model = Model(inputs=[words_input, casing_input, character_input, postag_input], outputs=[output])

        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=self.optimizer)

        self.init_weights = self.model.get_weights()

        plot_model(self.model, to_file='model' + self.run_name + '.png')

        print("Model built. Saved model.png\n")

    def train(self):
        """Default training"""

        self.f1_test_history = []
        self.f1_dev_history = []

        for epoch in range(self.epochs):
            print("Epoch {}/{}".format(epoch, self.epochs))
            for i, batch in enumerate(iterate_minibatches(self.train_batch, self.train_batch_len)):
                labels, tokens, casing, char, postag = batch
                self.model.train_on_batch([tokens, casing, char, postag], labels)

            # compute F1 scores
            predLabels, correctLabels = self.tag_dataset(self.test_batch, self.model)
            pre_test, rec_test, f1_test = compute_f1(predLabels, correctLabels, self.idx2Label)
            self.f1_test_history.append(f1_test)
            print("f1 test ", round(f1_test, 4))
            print("pre_test test ", round(pre_test, 4))
            print("recall1 test ", round(rec_test, 4))

            predLabels, correctLabels = self.tag_dataset(self.dev_batch, self.model)
            pre_dev, rec_dev, f1_dev = compute_f1(predLabels, correctLabels, self.idx2Label)
            self.f1_dev_history.append(f1_dev)
            print("f1 dev ", round(f1_dev, 4))
            print("pre_test dev ", round(pre_dev, 4))
            print("recall1 dev ", round(rec_dev, 4))

        print("Final F1 test score: ", f1_test)
        print("Final F1 dev score: ", f1_dev)

        print("Training finished.")

        # save model
        self.modelName = "{}_{}_{}_{}_{}_{}_{}".format(self.epochs,
                                                       self.dropout,
                                                       self.dropout_recurrent,
                                                       self.lstm_state_size,
                                                       self.conv_size,
                                                       self.learning_rate,
                                                       self.optimizer.__class__.__name__
                                                       )

        modelName = self.modelName + self.run_name + ".h5"
        self.model.save(modelName)
        print("Model weights saved.")

        self.model.set_weights(self.init_weights)  # clear model
        print("Model weights cleared.")

    def writeToFile(self):
        """Write output to file"""

        # .txt file format
        # [epoch  ]
        # [f1_test]
        # [f1_dev ]

        output = np.matrix([[int(i) for i in range(self.epochs)], self.f1_test_history, self.f1_dev_history])

        fileName = self.modelName + ".txt"
        with open(fileName, 'wb') as f:
            for line in output:
                np.savetxt(f, line, fmt='%.5f')

        print("Model performance written to file: " + self.modelName + ".txt")

    def save_data_to_pickle(self):
        with open('indexes/word2Idx.pkl', 'wb') as fp:
            pickle.dump(self.word2Idx_, fp)
        with open('indexes/label2Idx.pkl', 'wb') as fp:
            pickle.dump(self.label2Idx, fp)
        with open('indexes/idx2Label.pkl', 'wb') as fp:
            pickle.dump(self.idx2Label, fp)
        with open('indexes/char2Idx.pkl', 'wb') as fp:
            pickle.dump(self.char2Idx, fp)
        with open('indexes/case2Idx.pkl', 'wb') as fp:
            pickle.dump(self.case2Idx_, fp)
        with open('indexes/postag2Idx.pkl', 'wb') as fp:
            pickle.dump(self.postag2Idx, fp)
        with open('indexes/train_set.pkl', 'wb') as fp:
            pickle.dump(self.train_set, fp)
        with open('indexes/dev_set.pkl', 'wb') as fp:
            pickle.dump(self.dev_set, fp)
        with open('indexes/test_set.pkl', 'wb') as fp:
            pickle.dump(self.test_set, fp)

    print("Class initialised.")


def load_data_from_pickle():
    with open('indexes/word2Idx.pkl', 'rb') as fp:
        word2Idx = pickle.load(fp)
    with open('indexes/label2Idx.pkl', 'rb') as fp:
        label2Idx = pickle.load(fp)
    with open('indexes/idx2Label.pkl', 'rb') as fp:
        idx2Label = pickle.load(fp)
    with open('indexes/char2Idx.pkl', 'rb') as fp:
        char2Idx = pickle.load(fp)
    with open('indexes/case2Idx.pkl', 'rb') as fp:
        case2Idx = pickle.load(fp)
    with open('indexes/postag2Idx.pkl', 'rb') as fp:
        postag2Idx = pickle.load(fp)
    with open('indexes/train_set.pkl', 'rb') as fp:
        train_set = pickle.load(fp)
    with open('indexes/dev_set.pkl', 'rb') as fp:
        dev_set = pickle.load(fp)
    with open('indexes/test_set.pkl', 'rb') as fp:
        test_set = pickle.load(fp)

    with open('../main/train_complete_cleaned.pkl', 'rb') as fp:
        results_sentences_train_data = pickle.load(fp)
    with open('../main/dev_complete_cleaned.pkl', 'rb') as fp:
        results_sentences_dev_data = pickle.load(fp)
    with open('../main/test_complete_cleaned.pkl', 'rb') as fp:
        results_sentences_test_data = pickle.load(fp)

    results_sentences_test_scenario_main_data = []
    return word2Idx, label2Idx, idx2Label, char2Idx, case2Idx, postag2Idx, train_set, dev_set, test_set, results_sentences_train_data, results_sentences_dev_data, results_sentences_test_data, results_sentences_test_scenario_main_data


def execute_model(run_name):
    """Set parameters"""

    EPOCHS = 30  # paper: 80
    DROPOUT = 0.5  # paper: 0.68
    DROPOUT_RECURRENT = 0.25  # not specified in paper, 0.25 recommended
    LSTM_STATE_SIZE = 200  # paper: 275
    CONV_SIZE = 3  # paper: 3
    LEARNING_RATE = 0.0105  # paper 0.0105
    OPTIMIZER = Nadam()  # paper uses SGD(lr=self.learning_rate), Nadam() recommended

    """Construct and run model"""

    cnn_blstm = CNN_BLSTM(EPOCHS, DROPOUT, DROPOUT_RECURRENT, LSTM_STATE_SIZE, CONV_SIZE, LEARNING_RATE, OPTIMIZER)
    cnn_blstm.run_name = run_name
    cnn_blstm.loadData()
    cnn_blstm.addCharInfo()
    cnn_blstm.embed()
    cnn_blstm.createBatches()
    cnn_blstm.buildModel()
    cnn_blstm.train()

    cnn_blstm.writeToFile()

    # import matplotlib.pyplot as plt
    # plt.plot(cnn_blstm.f1_test_history, label = "F1 test")
    # plt.plot(cnn_blstm.f1_dev_history, label = "F1 dev")
    # plt.xlabel("Epochs")
    # plt.ylabel("F1 score")
    # plt.legend()
    # plt.show()


def load_model_scenario_2_dev(run_name):
    # model = load_model('30_0.5_0.25_200_3_0.0105_Nadam' + run_name + '.h5')
    model = load_model('30_0.5_0.25_200_3_0.0105_Nadamrun1.h5')
    word2Idx, label2Idx, idx2Label, char2Idx, case2Idx, postag2Idx, train_set, dev_set, test_set, results_sentences_train_data, results_sentences_dev_data, results_sentences_test_data, results_sentences_test_scenario_main_data = load_data_from_pickle()

    idx2Word = {v: k for k, v in word2Idx.items()}
    count = 0
    keyphrase = ''
    posible_nombre = ""
    posibles_abs = ""
    completo_posible_nombre = False
    for file, sentences in results_sentences_dev_data.items():
        correctLabels = []
        predLabels = []
        with open('../main/system/run1_dev/brat/subtask1/' + file.replace(".txt", "").replace(".xml", "") + '.ann', 'w') as f:
            keyphrases = []

            for sentence in sentences:
                word = []
                if len(sentence) > 0:
                    print("Count - sentence number: " + str(count))
                    print("In file: " + file)
                    tokens_, casing_, char_, labels, postags_ = dev_set[count]
                    tokens = np.asarray([tokens_])
                    casing = np.asarray([casing_])
                    char = np.asarray([char_])
                    postag = np.asarray([postags_])
                    pred = model.predict([tokens, casing, char, postag], verbose=True)[0]
                    pred = pred.argmax(axis=-1)  # Predict the classes
                    # correctLabels.append(labels)
                    predLabels.append(pred)

                    keyphrase = ''

                    abs = ''
                    print(tokens_)
                    id_word = 0
                    added = False
                    for num_w, word in enumerate(sentence):
                        if pred[id_word] != label2Idx['O']:  # label 'O' -> no label
                            label_predicted = idx2Label[pred[id_word]]
                            abs_pos = str(word[1]) + ";" + str(word[2])
                            if "B-" in idx2Label[pred[id_word]] or last_idx_label == label2Idx['O']:  # if last label were O or B-[target]
                                if "B-" in idx2Label[pred[id_word]]:  # B-[target] label
                                    if keyphrase != '':
                                        keyphrase = keyphrase.replace("'", "")
                                        keyphrases.append([keyphrase, abs, label])
                                        added = True
                            if "B-" in idx2Label[pred[id_word]]:  # B-[target] label
                                keyphrase = word[0]
                                abs = abs_pos
                                label = label_predicted.replace("B-", "")
                                added = False
                                if posible_nombre != "" and "NOMBRE_SUJETO_ASISTENCIA" in label:
                                    posibles_abs_ = posibles_abs
                                    posible_nombre_ = posible_nombre
                                    keyphrase = posible_nombre + " " + keyphrase
                                    abs = posibles_abs + ";" + abs
                                    posible_nombre = ""
                                    posibles_abs = ""
                                elif posible_nombre != "" and "" not in label:
                                    keyphrases.append([posible_nombre_, posibles_abs_, label])
                                    posible_nombre = ""
                                    posibles_abs = ""

                            elif "I-" in idx2Label[pred[id_word]]:
                                keyphrase = keyphrase + " " + word[0]
                                abs = abs + ";" + abs_pos
                                added = False
                                # if posible_nombre != "":
                                #     posible_nombre = ""
                                #     posibles_abs = ""
                                completo_posible_nombre = True
                        if num_w == len(sentence)-1:
                            if keyphrase != '' and not added:
                                if "NOMBRE_SUJETO_ASISTENCIA" in label and posible_nombre == "" and not completo_posible_nombre: # if next concept is the same concept...
                                    posible_nombre = keyphrase.replace("'", "")
                                    posibles_abs = abs
                                else:
                                    keyphrase = keyphrase.replace("'", "")
                                    keyphrases.append([keyphrase, abs, label])
                                    completo_posible_nombre = False
                        last_idx_label = pred[id_word]
                        id_word += 1
                    count += 1
            count_ = 1
            for kp in keyphrases:
                if "UNKNOWN_TOKEN" not in kp[0] and kp[1][0] != ";":
                    pos = kp[1].split(";")
                    text = "T" + str(count_) + "\t" + kp[2] + " " + str(pos[0]) + " " + str(pos[-1]) + "\t" + kp[0]
                    f.write("%s\n" % text)
                    count_ += 1
                if kp[1][0] == ";":
                    print("Error keyphrase")
                    print(kp)

def load_model_run2_test(run_name):
    # model = load_model('30_0.5_0.25_200_3_0.0105_Nadam' + run_name + '.h5')
    model = load_model('30_0.5_0.25_200_3_0.0105_Nadamrun1.h5')
    word2Idx, label2Idx, idx2Label, char2Idx, case2Idx, postag2Idx, train_set, dev_set, test_set, results_sentences_train_data, results_sentences_dev_data, results_sentences_test_data, results_sentences_test_scenario_main_data = load_data_from_pickle()

    idx2Word = {v: k for k, v in word2Idx.items()}
    count = 0
    keyphrase = ''
    posible_nombre = ""
    posibles_abs = ""
    completo_posible_nombre = False
    for file, sentences in results_sentences_test_data.items():
        correctLabels = []
        predLabels = []
        with open('../main/system/run2/brat/subtask1/' + file.replace(".txt", "").replace(".xml", "") + '.ann', 'w') as f:
            keyphrases = []

            for sentence in sentences:
                word = []
                if len(sentence) > 0:
                    print("Count - sentence number: " + str(count))
                    print("In file: " + file)
                    tokens_, casing_, char_, labels, postags_ = test_set[count]
                    tokens = np.asarray([tokens_])
                    casing = np.asarray([casing_])
                    char = np.asarray([char_])
                    postag = np.asarray([postags_])
                    pred = model.predict([tokens, casing, char, postag], verbose=True)[0]
                    pred = pred.argmax(axis=-1)  # Predict the classes
                    # correctLabels.append(labels)
                    predLabels.append(pred)

                    keyphrase = ''

                    abs = ''
                    print(tokens_)
                    id_word = 0
                    added = False
                    for num_w, word in enumerate(sentence):
                        if pred[id_word] != label2Idx['O']:  # label 'O' -> no label
                            label_predicted = idx2Label[pred[id_word]]
                            abs_pos = str(word[1]) + ";" + str(word[2])
                            if "B-" in idx2Label[pred[id_word]] or last_idx_label == label2Idx['O']:  # if last label were O or B-[target]
                                if "B-" in idx2Label[pred[id_word]]:  # B-[target] label
                                    if keyphrase != '':
                                        keyphrase = keyphrase.replace("'", "")
                                        keyphrases.append([keyphrase, abs, label])
                                        added = True
                            if "B-" in idx2Label[pred[id_word]]:  # B-[target] label
                                keyphrase = word[0]
                                abs = abs_pos
                                label = label_predicted.replace("B-", "")
                                added = False
                                if posible_nombre != "" and "NOMBRE_SUJETO_ASISTENCIA" in label:
                                    posibles_abs_ = posibles_abs
                                    posible_nombre_ = posible_nombre
                                    keyphrase = posible_nombre + " " + keyphrase
                                    abs = posibles_abs + ";" + abs
                                    posible_nombre = ""
                                    posibles_abs = ""
                                elif posible_nombre != "" and "" not in label:
                                    keyphrases.append([posible_nombre_, posibles_abs_, label])
                                    posible_nombre = ""
                                    posibles_abs = ""

                            elif "I-" in idx2Label[pred[id_word]]:
                                keyphrase = keyphrase + " " + word[0]
                                abs = abs + ";" + abs_pos
                                added = False
                                # if posible_nombre != "":
                                #     posible_nombre = ""
                                #     posibles_abs = ""
                                completo_posible_nombre = True
                        if num_w == len(sentence)-1:
                            if keyphrase != '' and not added:
                                if "NOMBRE_SUJETO_ASISTENCIA" in label and posible_nombre == "" and not completo_posible_nombre: # if next concept is the same concept...
                                    posible_nombre = keyphrase.replace("'", "")
                                    posibles_abs = abs
                                else:
                                    keyphrase = keyphrase.replace("'", "")
                                    keyphrases.append([keyphrase, abs, label])
                                    completo_posible_nombre = False
                        last_idx_label = pred[id_word]
                        id_word += 1
                    count += 1
            count_ = 1
            for kp in keyphrases:
                if "UNKNOWN_TOKEN" not in kp[0] and kp[1][0] != ";":
                    pos = kp[1].split(";")
                    text = "T" + str(count_) + "\t" + kp[2] + " " + str(pos[0]) + " " + str(pos[-1]) + "\t" + kp[0]
                    f.write("%s\n" % text)
                    count_ += 1
                if kp[1][0] == ";":
                    print("Error keyphrase")
                    print(kp)



def load_model_scenario_2_test(run_name):
    # model = load_model('30_0.5_0.25_200_3_0.0105_Nadam' + run_name + '.h5')
    model = load_model('30_0.5_0.25_200_3_0.0105_Nadamrun1.h5')
    word2Idx, label2Idx, idx2Label, char2Idx, case2Idx, postag2Idx, train_set, dev_set, test_set, results_sentences_train_data, results_sentences_dev_data, results_sentences_test_data, results_sentences_test_scenario_main_data = load_data_from_pickle()

    idx2Word = {v: k for k, v in word2Idx.items()}
    count = 0
    keyphrase = ''
    for file, sentences in results_sentences_test_data.items():
        correctLabels = []
        predLabels = []

        with open('../main/system/run1/brat/subtask1/' + file.replace(".txt", "") + '.ann', 'w') as f:
            keyphrases = []
            for sentence in sentences:
                word = []
                if len(sentence) > 0:
                    print("Count - sentence number: " + str(count))
                    print("In file: " + file)
                    tokens_, casing_, char_, labels, postags_ = test_set[count]
                    tokens = np.asarray([tokens_])
                    casing = np.asarray([casing_])
                    char = np.asarray([char_])
                    postag = np.asarray([postags_])
                    pred = model.predict([tokens, casing, char, postag], verbose=True)[0]
                    pred = pred.argmax(axis=-1)  # Predict the classes
                    # correctLabels.append(labels)
                    predLabels.append(pred)

                    keyphrase = ''
                    abs = ''
                    print(tokens_)
                    id_word = 0
                    added = False
                    for num_w, word in enumerate(sentence):
                        if pred[id_word] != label2Idx['O']:  # label 'O' -> no label
                            label_predicted = idx2Label[pred[id_word]]
                            abs_pos = str(word[1]) + ";" + str(word[2])
                            if "B-" in idx2Label[pred[id_word]] or last_idx_label == label2Idx['O']:  # if last label were O or B-[target]
                                if "B-" in idx2Label[pred[id_word]]:  # B-[target] label
                                    if keyphrase != '':
                                        keyphrase = keyphrase.replace("'", "")
                                        keyphrases.append([keyphrase, abs, label])
                                        added = True
                            if "B-" in idx2Label[pred[id_word]]:  # B-[target] label
                                keyphrase = word[0]
                                abs = abs_pos
                                label = label_predicted.replace("B-", "")
                                added = False
                            elif "I-" in idx2Label[pred[id_word]]:
                                keyphrase = keyphrase + " " + word[0]
                                abs = abs + ";" + abs_pos
                                added = False
                        if num_w == len(sentence) - 1:
                            if keyphrase != '' and not added:
                                keyphrase = keyphrase.replace("'", "")
                                keyphrases.append([keyphrase, abs, label])
                        last_idx_label = pred[id_word]
                        id_word += 1
                    count += 1
            count_ = 1
            for kp in keyphrases:
                if "UNKNOWN_TOKEN" not in kp[0] and kp[1][0] != ";":
                    pos = kp[1].split(";")
                    text = "T" + str(count_) + "\t" + kp[2] + " " + str(pos[0]) + " " + str(pos[-1]) + "\t" + kp[0]
                    f.write("%s\n" % text)
                    count_ += 1
                if kp[1][0] == ";":
                    print("Error keyphrase")
                    print(kp)


run_name = "run1_dev"
run_name = "run2"

# execute_model(run_name)

# load_model_scenario_2_test(run_name)
# load_model_scenario_2_dev(run_name)
# load_model_run2_test(run_name)



def clean_files():
    onlyfiles = [f for f in listdir("../dataset/background/") if isfile(join("../dataset/background/", f))]
    onlyfiles.sort()
    data = []
    data_w_file = {}
    for file in onlyfiles:
        # copyfile("../dataset/dev/brat/" + file.replace(".ann", "") + ".txt", "../main/system/run1_dev/brat/" + file.replace(".ann", "") + ".txt")#DEV
        # copyfile("../dataset/dev/brat/" + file.replace(".ann", "").replace(".txt", "") + ".txt", "../main/system/run1/brat/" + file.replace(".ann", "").replace(".txt", "") + ".txt") #DEV
        # copyfile("../dataset/background/" + file.replace(".ann", "").replace(".txt", "") + ".txt", "../main/system/run2/brat/subtask2/" + file.replace(".ann", "").replace(".txt", "") + ".txt")#TEST
        copyfile("../dataset/background/" + file.replace(".ann", "").replace(".txt", "") + ".txt", "../main/system/run2/brat/subtask1/" + file.replace(".ann", "").replace(".txt", "") + ".txt")#TEST

clean_files()

def add_gazzter():
    nombres = open("../main/Nombres (Firstnames).txt")
    apellidos = open("../main/Apellidos (Lastnames).txt")

    concepts = []
    postag = "PROPN"
    for nombre in nombres.readlines():
        for apellido in apellidos.readlines():
            concepts.append(nombre + " " + postag +  " " + "B-NOMBRE_SUJETO_ASISTENCIA")
            concepts.append(apellido + " " + postag +  " " + "I-NOMBRE_SUJETO_ASISTENCIA")

            concepts.append(nombre + " " + postag +  " " + "B-NOMBRE_PERSONAL_SANITARIO")
            concepts.append(apellido + " " + postag +  " " + "I-NOMBRE_PERSONAL_SANITARIO")

    lugar = open("../main/Apellidos (Lastnames).txt")