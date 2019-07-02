# import xml.etree.ElementTree as ET
import pickle
import re

from main.postagger import PosTaggerStanford
from main.classes import i2b2Annotation
# from Med_Tagger.Med_Tagger import Med_Tagger

from os import listdir
from os.path import isfile, join

DATASET_PATH_TRAIN_XML = "../dataset/train/xml/"
DATASET_PATH_DEV_XML = "../dataset/dev/xml/"
DATASET_PATH_TEST_TXT = "../dataset/background/"
FILE_INPUT_NN_TRAIN = "../main/neuralnetwork_input/train.txt"
FILE_INPUT_NN_DEV = "../main/neuralnetwork_input/dev.txt"
FILE_INPUT_NN_TEST = "../main/neuralnetwork_input/test.txt"

FILE_SENTENCES_TRAIN_COMPLETED = '../main/train_complete.pkl'
FILE_SENTENCES_DEV_COMPLETED = '../main/dev_complete.pkl'
FILE_SENTENCES_TEST_COMPLETED = '../main/test_complete.pkl'

FILE_SENTENCES_TRAIN_CLEANED = '../main/train_complete_cleaned.pkl'
FILE_SENTENCES_DEV_CLEANED = '../main/dev_complete_cleaned.pkl'
FILE_SENTENCES_TEST_CLEANED = '../main/test_complete_cleaned.pkl'

FILE_SENTENCES_TRAIN_COMPLETED_NOT_CLEANED = '../main/train_complete_not_cleaned.pkl'
FILE_SENTENCES_DEV_COMPLETED_NOT_CLEANED = '../main/dev_complete_not_cleaned.pkl'
FILE_SENTENCES_TEST_COMPLETED_NOT_CLEANED = '../main/test_complete_not_cleaned.pkl'

CACHE_SPANS = False
CACHE_ANNOTATED_DATA = False


def calculate_spans(gold_annotations_text, isTest=False):
    if isTest:
        sentences = gold_annotations_text
    else:
        sentences = gold_annotations_text.split("\n")
    pos_ini = 0
    sents = []
    for sentence in sentences:
        spans = []
        sentence = sentence.replace(":", " ")
        words = sentence.split(" ")
        for w in words:
            word = w.strip("(").strip("(").strip(".").strip(",").strip("(").strip('"').strip(':').strip(';')
            pos_ini_word = pos_ini
            pos_end_word = pos_ini_word + len(word)
            spans.append([word, pos_ini_word, pos_end_word])
            pos_ini = pos_ini_word + len(w) + 1
        sents.append(spans)
    return sents


def annotate_with_gold(sentences_spans, gold_annotations, isTest=False):
    for sent in sentences_spans:
        i = 0
        for span in sent:
            span_word = span[0]
            span_ini = span[1]
            span_end = span[2]
            found_label = False
            if len(span_word) != 0 and span_word != " ":
                if re.search('[a-zA-Z0-9]', span_word):
                    postag = PosTaggerStanford.post_tag_word(span_word)
                    sent[i].extend([postag])

                else:
                    sent[i].extend(["X"])
            else:
                sent[i].extend(["X"])
            if not isTest:
                for phi in gold_annotations.get_phi():
                    if span_ini == int(phi.start) and span_end == int(phi.end):
                        label = 'B-' + phi.TYPE
                        sent[i].extend([label])
                        found_label = True
                        break
                    elif span_ini == int(phi.start) and int(phi.end) > span_end:
                        label = 'B-' + phi.TYPE
                        sent[i].extend([label])
                        found_label = True
                        break
                    elif int(phi.start) < span_ini and int(phi.end) >= span_end:
                        label = 'I-' + phi.TYPE
                        sent[i].extend([label])
                        found_label = True
                        break
            if not found_label:
                label = 'O'
                # sentences_spans[i].extend(["X"])
                sent[i].extend([label])
            i += 1
    return sentences_spans


def format_input_file(file_name):
    gold_annotations = i2b2Annotation(file_name=file_name)
    sentences_spans = calculate_spans(gold_annotations.text)
    sentences_spans_annotated_with_gold = annotate_with_gold(sentences_spans, gold_annotations)
    return sentences_spans_annotated_with_gold


def format_input_file_test(file_name):  # todo fix
    gold_annotations = open(file_name)
    text = []
    for line in gold_annotations.readlines():
        if len(line) > 0:
            text.append(line.replace("\n", ""))
    sentences_spans = calculate_spans(text, isTest=True)
    sentences_spans_annotated_with_gold = annotate_with_gold(sentences_spans, text, isTest=True)
    return sentences_spans_annotated_with_gold


def output_train():
    onlyfiles = [f for f in listdir(DATASET_PATH_TRAIN_XML) if isfile(join(DATASET_PATH_TRAIN_XML, f))]
    onlyfiles.sort()
    data = []
    for file in onlyfiles:
        sentences_spans_annotated_with_gold = format_input_file(DATASET_PATH_TRAIN_XML + file)
        data.append(sentences_spans_annotated_with_gold)
    # write_input_file_to_nn(data, FILE_INPUT_NN_TRAIN, isTest=False)
    with open(FILE_SENTENCES_TRAIN_COMPLETED, 'wb') as fp:
        pickle.dump(data, fp)


def output_dev():
    onlyfiles = [f for f in listdir(DATASET_PATH_DEV_XML) if isfile(join(DATASET_PATH_DEV_XML, f))]
    onlyfiles.sort()
    data = []
    data_w_file = {}
    for file in onlyfiles:
        sentences_spans_annotated_with_gold = format_input_file(DATASET_PATH_DEV_XML + file)
        data.append(sentences_spans_annotated_with_gold)
        data_w_file[file] = sentences_spans_annotated_with_gold
    # write_input_file_to_nn(data, FILE_INPUT_NN_DEV, isTest=False)
    with open(FILE_SENTENCES_DEV_COMPLETED, 'wb') as fp:
        pickle.dump(data_w_file, fp)


def output_test():
    onlyfiles = [f for f in listdir(DATASET_PATH_TEST_TXT) if isfile(join(DATASET_PATH_TEST_TXT, f))]
    onlyfiles.sort()
    data = []
    data_w_file = {}
    for file in onlyfiles:
        sentences_spans_annotated_with_gold = format_input_file_test(DATASET_PATH_TEST_TXT + file)
        data.append(sentences_spans_annotated_with_gold)
        data_w_file[file] = sentences_spans_annotated_with_gold
    # write_input_file_to_nn(data, FILE_INPUT_NN_TEST, isTest=True)
    with open(FILE_SENTENCES_TEST_COMPLETED, 'wb') as fp:
        pickle.dump(data_w_file, fp)


def clean_input_file(file_input, file_not_cleaned, file_output, sentences_cleaneeed, isTest=False):
    with open(file_input, 'rb') as fp:
        data = pickle.load(fp)
    with open(file_input, 'rb') as fp:
        output_clean = pickle.load(fp)

    # count_doc = 0
    for file_, doc in data.items():
        count_sent = 0
        for sentence in doc:
            count_word = 0
            if len(sentence) > 51:
                output_clean[file_].pop(count_sent)
            elif len(sentence) == 0:
                output_clean[file_].pop(count_sent)
            else:
                for word in sentence:
                    if len(word) != 5:
                        print("ERROR")
                    w = word[0]
                    pos_ini = word[1]
                    pos_end = word[2]
                    postag = word[3]

                    output_clean[file_][count_sent][count_word][0] = word[0].replace(u"\xa0", "").replace(u"\x99", "").replace(u"\x89", "").replace(u"\x9c", "").replace(u"\x80", "").replace(u"\x9d", "").replace(u"\u200b", "").replace(u"\xad", "").replace(u"\x89", "").replace(u"\x96", "").replace(u"\x85", "").replace(u"\x99", "").replace(u"\xa0", "").replace(u"\ufeff", "").replace(u"\x82", "")
                    w = w.replace(u"\xa0", "").replace(u"\x99", "").replace(u"\x89", "").replace(u"\x9c", "").replace(u"\x80", "").replace(u"\x9d", "").replace(u"\u200b", "").replace(u"\xad", "").replace(u"\x89", "").replace(u"\x96", "").replace(u"\x85", "").replace(u"\x99", "").replace(u"\xa0", "").replace(u"\ufeff", "").replace(u"\x82", "")

                    # contains_alphanumeric_chars = re.search('[a-zA-Z]', postag)
                    # if len(postag) == 0 or postag == " " or not contains_alphanumeric_chars or contains_alphanumeric_chars is None:
                    # if len(postag) == 0 or postag == " ":
                    #     if postag != '_':
                    #         print("ERROR")
                    label = word[4]

                    # contains_alphanumeric_chars = re.search('[a-zA-Z]', label)
                    # if len(label) == 0 or label == " " or not contains_alphanumeric_chars or contains_alphanumeric_chars is None:
                    # if len(label) == 0 or label == " ":
                    #     print("ERROR")

                    contains_alphanumeric_chars = re.search('[a-zA-Z]', w)
                    # if len(w) == 0 or w == " " or not contains_alphanumeric_chars or contains_alphanumeric_chars is None:
                    if len(w) == 0 or w == " ":
                        output_clean[file_][count_sent].pop(count_word)
                    else:
                        count_word += 1
                count_sent += 1
        # count_doc += 1

    with open(sentences_cleaneeed, 'wb') as fp:
        pickle.dump(output_clean, fp)

    with open(file_not_cleaned, 'wb') as fp:
        pickle.dump(data, fp)

    with open(file_output, 'w') as f:
        f.write("%s\n" % "-DOCSTART- -X- -X- O\n")
        for file, sentence in output_clean.items():
            for w in sentence:
                for d in w:
                    text = d[0] + " " + d[3] + " " + d[4] + " " + 'O'
                    f.write("%s\n" % text)
                f.write("\n")

    with open(file_output.replace(".txt", "") + "_with_lines.txt", 'w') as f:
        f.write("%s\n" % "-DOCSTART- -X- -X- O\n")
        for file, sentence in output_clean.items():
            for w in sentence:
                for d in w:
                    text = d[0] + " " + d[3] + " " + d[4] + " " + 'O'
                    f.write("%s\n" % text)
                f.write("\n")
            f.write("EOF_" + file + "\n")


# output_train()
# output_dev()
output_test()

# clean_input_file(FILE_SENTENCES_TRAIN_COMPLETED, FILE_SENTENCES_TRAIN_COMPLETED_NOT_CLEANED, FILE_INPUT_NN_TRAIN, FILE_SENTENCES_TRAIN_CLEANED, isTest=False)
# clean_input_file(FILE_SENTENCES_DEV_COMPLETED, FILE_SENTENCES_DEV_COMPLETED_NOT_CLEANED, FILE_INPUT_NN_DEV, FILE_SENTENCES_DEV_CLEANED, isTest=False)
clean_input_file(FILE_SENTENCES_TEST_COMPLETED, FILE_SENTENCES_TEST_COMPLETED_NOT_CLEANED, FILE_INPUT_NN_TEST, FILE_SENTENCES_TEST_CLEANED, isTest=True)


# def prueba():
#     with open("../main/dev_complete_not_cleaned.pkl", 'rb') as fp:
#         file_ = pickle.load(fp)
#     with open('../main/dev_complete_cleaned.pkl', 'rb') as fp:
#         file_2 = pickle.load(fp)
#
#     onlyfiles = [f for f in listdir(DATASET_PATH_DEV_XML) if isfile(join(DATASET_PATH_DEV_XML, f))]
#     onlyfiles.sort()
#     data = {}
#     len_tot = 0
#
#     for file in onlyfiles:
#         gold_annotations = i2b2Annotation(file_name=DATASET_PATH_DEV_XML + file)
#         sentences = gold_annotations.text.split("\n")
#         data[file] = len(sentences)
#         len_tot += len(sentences)
#
#     other_count = 0
#     for doc in file_2:
#         count_sent = 0
#         for sent in doc:
#             if len(sent) > 0:
#                 other_count += 1
#             count_sent += 1
#     # print(data)
#     print(len_tot)

    # for f in file_:
    #
    #     print(len(f))


# def prueba():
#     char = " 0123456789abcdefghijklmno∧pq†rstuvwxyzAÂB²©CDEFηGλØÒHIJKιLMN´kOPâ≥QRSTãκUVWXYZäáéèê¼ΥíìóøöúüΔñÑÁÉªÍÓÚ–.,-ô_()[]’{}!?:;#'\"/\\%$`→&=*³+@^ò~|<δ>α°ºë®μµç×½•«¡!»β±ß¥àÇ™χ≈“≤”—γå¿?●ŀ′·"
#     s = set()
#     print(len(char))
#     for c in char:
#         s.add(c)
#
#     print(len(s))
#     text = ""
#     for c in s:
#         text = text + c
#     print(text)
#
#     if " " in text:
#         print("tap")
# prueba()
