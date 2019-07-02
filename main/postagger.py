import pickle

import nltk
import os


class PosTaggerStanford:

    sentences = [] #important, sentences are OBJECTS from utils.py
    postags = {}

    def __init__(self):
        self.tagger = nltk.CoreNLPParser('http://localhost:9003', tagtype='pos')
        # self.tagger = nltk.CoreNLPParser('http://localhost:9003', tagtype='ner')


    def post_tag_sentences(self, sents, getPickle=True, savePickle=True):
        if getPickle:
            if os.path.isfile('./aluned/postags.pkl'):
                self.load_pickle()
        else:
            self.postags = {}
            self.sentences = sents
            for sentence in self.sentences:
                # sents.append(tagger.api_call(sentence.text))
                self.postags[sentence.text] = self.tagger.tag([sentence.text])

            if savePickle:
                self.save_pickle()

        return self.postags

    @staticmethod
    def post_tag_word(word):
        tagger = nltk.CoreNLPParser('http://localhost:9003', tagtype='pos')
        post_tag = tagger.tag([word])
        tag = post_tag[0][1]
        return tag

    def complete_tagger(self, text, tags='tokenize,ssplit,pos,lemma,ner,depparse,coref'):
        self.tagger = nltk.CoreNLPParser(url='http://localhost:9003')
        self.tagger.parser_annotator = tags
        return self.tagger.api_call(text)

    def save_pickle(self):
        with open('./aluned/postags.pkl', 'wb') as fp:
            pickle.dump(self.postags, fp)

        with open("./aluned/postags.txt", "w") as output:
            output.write(str(self.postags))

    def load_pickle(self):
        with open('./aluned/postags.pkl', 'rb') as fp:
            self.postags = pickle.load(fp)