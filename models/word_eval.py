import numpy as np
import scipy.stats as st
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize
from preprocess import get_tokenizer

"""
Word similarity test
Inspired by:
https://github.com/dongjun-Lee/kor2vec/blob/master/test/similarity_test.py
"""
class WordSimilarityCheck:

    def __init__(self, vecs_fname, method="word2vec", dim=128, tokenize=True):
        if tokenize:
            self.tokenizer = get_tokenizer("mecab")
        else:
            self.tokenizer = None
        self.tokenize = tokenize
        self.dim = dim
        self.dictionary, self.words, self.vecs = self.load_vectors(vecs_fname, method)

    def load_vectors(self, vecs_fname, method):
        if method == "word2vec":
            model = Word2Vec.load(vecs_fname)
            words = model.wv.index2word
            vecs = model.wv.vectors
        else:
            words, vecs = [], []
            with open(vecs_fname, 'r', encoding='utf-8') as f:
                if method == "fasttext":
                    next(f)  # skip head line
                for line in f:
                    if method == "swivel":
                        splited_line = line.replace("\n", "").strip().split("\t")
                    else:
                        splited_line = line.replace("\n", "").strip().split(" ")
                    words.append(splited_line[0])
                    vec = [float(el) for el in splited_line[1:]]
                    vecs.append(vec)
        unit_vecs = normalize(vecs, norm='l2', axis=1)
        dictionary = {}
        for word, vec in zip(words, unit_vecs):
            dictionary[word] = vec
        return dictionary, words, unit_vecs

    def compute_total_cosine(self, query, topn=10):
        query_vec = self.get_sentence_vector(query)
        query_unit_vec = query_vec / np.linalg.norm(query_vec)
        scores = np.dot(self.vecs, query_unit_vec)
        print(scores)
        return sorted(zip(self.words, scores), key=lambda x: x[1], reverse=True)[:topn]

    def word_sim_test(self, test_fname):
        actual_sim_list, pred_sim_list = [], []
        missed = 0
        with open(test_fname, 'r') as pairs:
            for pair in pairs:
                w1, w2, actual_sim = pair.strip().split(",")
                try:
                    w1_vec = self.get_sentence_vector(w1)
                    w2_vec = self.get_sentence_vector(w2)
                    score = np.dot(w1_vec, w2_vec)
                    actual_sim_list.append(float(actual_sim))
                    pred_sim_list.append(score)
                except KeyError:
                    missed += 1
        spearman, _ = st.spearmanr(actual_sim_list, pred_sim_list)
        pearson, _ = st.pearsonr(actual_sim_list, pred_sim_list)
        print(spearman, pearson, missed)

    def get_sentence_vector(self, sentence):
        if self.tokenize:
            tokens = self.tokenizer.morphs(sentence)
        else:
            tokens = sentence.split(" ")
        token_vecs = []
        for token in tokens:
            if token in self.dictionary.keys():
                token_vecs.append(self.dictionary[token])
        if len(token_vecs) > 0:
            result = np.mean(token_vecs, axis=1)
            print(token_vecs)
            print(result)
        else:
            result = np.zeros(self.dim)
        return result


model = WordSimilarityCheck("data/word2vec.vecs", "word2vec", dim=128, tokenize=True)
model.word_sim_test("data/kor_ws353.csv") # 0.07967824412220588 0.052695494326999485 0

model = WordSimilarityCheck("data/glove.vecs.txt", "glove", dim=128, tokenize=True)
model.word_sim_test("data/kor_ws353.csv") # 0.04503284244559433 0.052921610272604946 0

model = WordSimilarityCheck("data/fasttext.vecs.vec", "fasttext", dim=100, tokenize=True)
model.word_sim_test("data/kor_ws353.csv") # 0.10824313907957996 0.10782498999451953 0

model = WordSimilarityCheck("data/swivel.vecs/row_embedding.tsv", "swivel", dim=128, tokenize=True)
model.word_sim_test("data/kor_ws353.csv") # 0.09000367565864958 0.11364630219391056 0