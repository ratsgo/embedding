import sys
import numpy as np
import scipy.stats as st
from gensim.models import Word2Vec
from fasttext import load_model as load_ft_model
from sklearn.preprocessing import normalize

from soynlp.hangle import compose, character_is_korean
from preprocess import get_tokenizer, jamo_sentence

sys.path.append('models')
from visualize_utils import visualize_words, visualize_between_words


class WordEmbeddingEvaluator:

    def __init__(self, vecs_txt_fname, vecs_bin_fname=None, method="word2vec", dim=100, tokenizer_name="mecab"):
        self.tokenizer = get_tokenizer(tokenizer_name)
        self.tokenizer_name = tokenizer_name
        self.dim = dim
        self.method = method
        self.dictionary, self.words, self.vecs = self.load_vectors(vecs_txt_fname, method)
        if "fasttext" in method:
            self.model = load_ft_model(vecs_bin_fname)

    def load_vectors(self, vecs_fname, method):
        if method == "word2vec":
            model = Word2Vec.load(vecs_fname)
            words = model.wv.index2word
            vecs = model.wv.vectors
        else:
            words, vecs = [], []
            with open(vecs_fname, 'r', encoding='utf-8') as f:
                if "fasttext" in method:
                    next(f)  # skip head line
                for line in f:
                    if method == "swivel":
                        splited_line = line.strip().split("\t")
                    else:
                        splited_line = line.strip().split(" ")
                    words.append(splited_line[0])
                    vec = [float(el) for el in splited_line[1:]]
                    vecs.append(vec)
        unit_vecs = normalize(vecs, norm='l2', axis=1)
        dictionary = {}
        for word, vec in zip(words, unit_vecs):
            dictionary[word] = vec
        return dictionary, words, unit_vecs

    def get_word_vector(self, word):
        if self.method == "fasttext-jamo":
            word = jamo_sentence(word)
        if self._is_in_vocabulary(word):
            vector = self.dictionary[word]
        else:
            if "fasttext" in self.method:
                vector = self.model.get_word_vector(word)
            else:
                vector = np.zeros(self.dim)
        return vector

    # token vector들을 lookup한 뒤 평균을 취한다
    def get_sentence_vector(self, sentence):
        if self.tokenizer_name == "khaiii":
            tokens = []
            for word in self.tokenizer.analyze(sentence):
                tokens.extend([str(m).split("/")[0] for m in word.morphs])
        else:
            tokens = self.tokenizer.morphs(sentence)
        token_vecs = []
        for token in tokens:
            token_vecs.append(self.get_word_vector(token))
        return np.mean(token_vecs, axis=0)

    def _is_in_vocabulary(self, word):
        if self.method == "fasttext-jamo":
            word = jamo_sentence(word)
        return word in self.dictionary.keys()

    def most_similar(self, query, topn=10):
        query_vec = self.get_sentence_vector(query)
        return self.most_similar_by_vector(query_vec, topn)

    def most_similar_by_vector(self, query_vec, topn=10):
        query_vec_norm = np.linalg.norm(query_vec)
        if query_vec_norm != 0:
            query_unit_vec = query_vec / query_vec_norm
        else:
            query_unit_vec = query_vec
        scores = np.dot(self.vecs, query_unit_vec)
        topn_candidates = sorted(zip(self.words, scores), key=lambda x: x[1], reverse=True)[1:topn+1]
        if self.method == "fasttext-jamo":
            return [(self.jamo_to_word(word), score) for word, score in topn_candidates]
        else:
            return topn_candidates

    def jamo_to_word(self, jamo):
        jamo_list, idx = [], 0
        while idx < len(jamo):
            if not character_is_korean(jamo[idx]):
                jamo_list.append(jamo[idx])
                idx += 1
            else:
                jamo_list.append(jamo[idx:idx + 3])
                idx += 3
        word = ""
        for jamo_char in jamo_list:
            if len(jamo_char) == 1:
                word += jamo_char
            elif jamo_char[2] == "-":
                word += compose(jamo_char[0], jamo_char[1], " ")
            else:
                word += compose(jamo_char[0], jamo_char[1], jamo_char[2])
        return word

    """
    Word similarity test
    Inspired by:
    https://github.com/dongjun-Lee/kor2vec/blob/master/test/similarity_test.py
    """
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
        print("spearman corr:", spearman, ", pearson corr:", pearson, ", # of errors:", missed)

    """
    Word Analogy test
    Inspired by:
    https://github.com/dongjun-Lee/kor2vec/blob/master/test/analogy_test.py
    """
    def word_analogy_test(self, test_fname, topn=30, verbose=False):
        correct, total, missed = 0, 0, 0
        with open(test_fname, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith("#") or len(line) <= 1:
                    continue
                words = line.strip().split(" ")
                query_vecs = self.get_analogy_vector(words[:-1])
                try:
                    word_with_scores = self.most_similar_by_vector(query_vecs, topn)
                    if verbose:
                        print(words[0] + " - " + words[1] + " + " + words[2])
                        print("correct answer:", words[3])
                        print("predicted answers:", word_with_scores)
                        print("")
                    similar_words = [el[0] for el in word_with_scores]
                    if words[-1] in similar_words:
                        correct += 1
                except:
                    missed += 1
                total += 1
        print("# of correct answer:", correct, ", # of data:", total, ", # of errors:", missed)

    def get_analogy_vector(self, words):
        if len(words) == 3:
            token_1 = self.get_sentence_vector(words[0])
            token_2 = self.get_sentence_vector(words[1])
            token_3 = self.get_sentence_vector(words[2])
            result = token_2 + token_3 - token_1
        else:
            result = np.zeros(self.dim)
        return result

    """
    Visualize word representions with T-SNE, Bokeh
    Inspired by:
    https://www.kaggle.com/yohanb/t-sne-bokeh
    https://bokeh.pydata.org
    """
    def visualize_words(self, words_fname, palette="Viridis256"):
        words = set()
        for line in open(words_fname, 'r', encoding='utf-8'):
            if not line.startswith("#"):
                for word in line.strip().split(" "):
                    if len(word) > 0:
                        words.add(word)
        vecs = np.array([self.get_sentence_vector(word) for word in words])
        visualize_words(words, vecs, palette)

    def visualize_between_words(self, words_fname, palette="Viridis256"):
        words = set()
        for line in open(words_fname, 'r'):
            if not line.startswith("#"):
                for word in line.strip().split(" "):
                    if len(word) > 0:
                        words.add(word)
        vecs = [self.get_sentence_vector(word) for word in words]
        visualize_between_words(words, vecs, palette)
