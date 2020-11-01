import argparse, os, sys, math
import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import TruncatedSVD
from soynlp.word import pmi
from soynlp.vectorizer import sent_to_word_contexts_matrix
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocess import get_tokenizer


class Word2VecCorpus:

    def __init__(self, corpus_fname):
        self.corpus_fname = corpus_fname

    def __iter__(self):
        with open(self.corpus_fname, 'r') as f:
            for sentence in f:
                tokens = sentence.replace('\n', '').strip().split(" ")
                yield tokens


def train_word2vec(corpus_fname, model_fname):
    make_save_path(model_fname)
    corpus = Word2VecCorpus(corpus_fname)
    model = Word2Vec(corpus, size=100, workers=4, sg=1)
    model.save(model_fname)

"""
Latent Semantic Analysis
Inspired by:
https://lovit.github.io/nlp/2018/04/22/context_vector_for_word_similarity
https://lovit.github.io/nlp/2018/04/22/implementing_pmi_numpy_practice
"""
def latent_semantic_analysis(corpus_fname, output_fname):
    make_save_path(output_fname)
    corpus = [sent.replace('\n', '').strip() for sent in open(corpus_fname, 'r').readlines()]
    # construct co-occurrence matrix (=word_context)
    # dynamic weight if True. co-occurrence weight = [1, (w-1)/w, (w-2)/w, ... 1/w]
    input_matrix, idx2vocab = sent_to_word_contexts_matrix(
        corpus,
        windows=3,
        min_tf=10,
        dynamic_weight=True,
        verbose=True)
    # compute truncated SVD
    cooc_svd = TruncatedSVD(n_components=100)
    cooc_vecs = cooc_svd.fit_transform(input_matrix)
    with open(output_fname + "-cooc.vecs", 'w') as f1:
        for word, vec in zip(idx2vocab, cooc_vecs):
            str_vec = [str(el) for el in vec]
            f1.writelines(word + ' ' + ' '.join(str_vec) + "\n")
    # Shift PPMI at k=0, (equal PPMI)
    # pmi(word, contexts)
    # px: Probability of rows(items)
    # py: Probability of columns(features)
    pmi_matrix, _, _ = pmi(input_matrix, min_pmi=math.log(5))
    # compute truncated SVD
    pmi_svd = TruncatedSVD(n_components=100)
    pmi_vecs = pmi_svd.fit_transform(input_matrix)
    with open(output_fname + "-pmi.vecs", 'w') as f2:
        for word, vec in zip(idx2vocab, pmi_vecs):
            str_vec = [str(el) for el in vec]
            f2.writelines(word + ' ' + ' '.join(str_vec) + "\n")


class CBoWModel(object):

    def __init__(self, train_fname, embedding_fname, model_fname, embedding_corpus_fname,
                 embedding_method="fasttext", is_weighted=True, average=False, dim=100, tokenizer_name="mecab"):
        # configurations
        make_save_path(model_fname)
        self.dim = dim
        self.average = average
        if is_weighted:
            model_full_fname = model_fname + "-weighted"
        else:
            model_full_fname = model_fname + "-original"
        self.tokenizer = get_tokenizer(tokenizer_name)
        if is_weighted:
            # ready for weighted embeddings
            self.embeddings = self.load_or_construct_weighted_embedding(embedding_fname, embedding_method, embedding_corpus_fname)
            print("loading weighted embeddings, complete!")
        else:
            # ready for original embeddings
            words, vectors = self.load_word_embeddings(embedding_fname, embedding_method)
            self.embeddings = defaultdict(list)
            for word, vector in zip(words, vectors):
                self.embeddings[word] = vector
            print("loading original embeddings, complete!")
        if not os.path.exists(model_full_fname):
            print("train Continuous Bag of Words model")
            self.model = self.train_model(train_fname, model_full_fname)
        else:
            print("load Continuous Bag of Words model")
            self.model = self.load_model(model_full_fname)

    def evaluate(self, test_data_fname, batch_size=3000, verbose=False):
        print("evaluation start!")
        test_data = self.load_or_tokenize_corpus(test_data_fname)
        data_size = len(test_data)
        num_batches = int((data_size - 1) / batch_size) + 1
        eval_score = 0
        for batch_num in range(num_batches):
            batch_sentences = []
            batch_tokenized_sentences = []
            batch_labels = []
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            features = test_data[start_index:end_index]
            for feature in features:
                sentence, tokens, label = feature
                batch_sentences.append(sentence)
                batch_tokenized_sentences.append(tokens)
                batch_labels.append(label)
            preds, curr_eval_score = self.predict_by_batch(batch_tokenized_sentences, batch_labels)
            eval_score += curr_eval_score
        if verbose:
            for sentence, pred, label in zip(batch_sentences, preds, batch_labels):
                print(sentence, ", pred:", pred, ", label:", label)
        print("# of correct:", str(eval_score), ", total:", str(len(test_data)), ", score:", str(eval_score / len(test_data)))

    def predict(self, sentence):
        tokens = self.tokenizer.morphs(sentence)
        sentence_vector = self.get_sentence_vector(tokens)
        scores = np.dot(self.model["vectors"], sentence_vector)
        pred = self.model["labels"][np.argmax(scores)]
        return pred

    def predict_by_batch(self, tokenized_sentences, labels):
        sentence_vectors, eval_score = [], 0
        for tokens in tokenized_sentences:
            sentence_vectors.append(self.get_sentence_vector(tokens))
        scores = np.dot(self.model["vectors"], np.array(sentence_vectors).T)
        preds = np.argmax(scores, axis=0)
        for pred, label in zip(preds, labels):
            if self.model["labels"][pred] == label:
                eval_score += 1
        return preds, eval_score

    def get_sentence_vector(self, tokens):
        vector = np.zeros(self.dim)
        for token in tokens:
            if token in self.embeddings.keys():
                vector += self.embeddings[token]
        if not self.average:
            vector /= len(tokens)
        vector_norm = np.linalg.norm(vector)
        if vector_norm != 0:
            unit_vector = vector / vector_norm
        else:
            unit_vector = np.zeros(self.dim)
        return unit_vector

    def load_or_tokenize_corpus(self, fname):
        data = []
        if os.path.exists(fname + "-tokenized"):
            with open(fname + "-tokenized", "r") as f1:
                for line in f1:
                    sentence, tokens, label = line.strip().split("\u241E")
                    data.append([sentence, tokens.split(), label])
        else:
            with open(fname, "r") as f2, open(fname + "-tokenized", "w") as f3:
                for line in f2:
                    sentence, label = line.strip().split("\u241E")
                    tokens = self.tokenizer.morphs(sentence)
                    data.append([sentence, tokens, label])
                    f3.writelines(sentence + "\u241E" + ' '.join(tokens) + "\u241E" + label + "\n")
        return data

    def compute_word_frequency(self, embedding_corpus_fname):
        total_count = 0
        words_count = defaultdict(int)
        with open(embedding_corpus_fname, "r") as f:
            for line in f:
                tokens = line.strip().split()
                for token in tokens:
                    words_count[token] += 1
                    total_count += 1
        return words_count, total_count

    def load_word_embeddings(self, vecs_fname, method):
        if method == "word2vec":
            model = Word2Vec.load(vecs_fname)
            words = model.wv.index2word
            vecs = model.wv.vectors
        else:
            words, vecs = [], []
            with open(vecs_fname, 'r', encoding='utf-8') as f1:
                if "fasttext" in method:
                    next(f1)  # skip head line
                for line in f1:
                    if method == "swivel":
                        splited_line = line.replace("\n", "").strip().split("\t")
                    else:
                        splited_line = line.replace("\n", "").strip().split(" ")
                    words.append(splited_line[0])
                    vec = [float(el) for el in splited_line[1:]]
                    vecs.append(vec)
        return words, vecs

    def load_or_construct_weighted_embedding(self, embedding_fname, embedding_method, embedding_corpus_fname, a=0.0001):
        dictionary = {}
        if os.path.exists(embedding_fname + "-weighted"):
            # load weighted word embeddings
            with open(embedding_fname + "-weighted", "r") as f2:
                for line in f2:
                    word, weighted_vector = line.strip().split("\u241E")
                    weighted_vector = [float(el) for el in weighted_vector.split()]
                    dictionary[word] = weighted_vector
        else:
            # load pretrained word embeddings
            words, vecs = self.load_word_embeddings(embedding_fname, embedding_method)
            # compute word frequency
            words_count, total_word_count = self.compute_word_frequency(embedding_corpus_fname)
            # construct weighted word embeddings
            with open(embedding_fname + "-weighted", "w") as f3:
                for word, vec in zip(words, vecs):
                    if word in words_count.keys():
                        word_prob = words_count[word] / total_word_count
                    else:
                        word_prob = 0.0
                    weighted_vector = (a / (word_prob + a)) * np.asarray(vec)
                    dictionary[word] = weighted_vector
                    f3.writelines(word + "\u241E" + " ".join([str(el) for el in weighted_vector]) + "\n")
        return dictionary

    def train_model(self, train_data_fname, model_fname):
        model = {"vectors": [], "labels": [], "sentences": []}
        train_data = self.load_or_tokenize_corpus(train_data_fname)
        with open(model_fname, "w") as f:
            for sentence, tokens, label in train_data:
                tokens = self.tokenizer.morphs(sentence)
                sentence_vector = self.get_sentence_vector(tokens)
                model["sentences"].append(sentence)
                model["vectors"].append(sentence_vector)
                model["labels"].append(label)
                str_vector = " ".join([str(el) for el in sentence_vector])
                f.writelines(sentence + "\u241E" + " ".join(tokens) + "\u241E" + str_vector + "\u241E" + label + "\n")
        return model

    def load_model(self, model_fname):
        model = {"vectors": [], "labels": [], "sentences": []}
        with open(model_fname, "r") as f:
            for line in f:
                sentence, _, vector, label = line.strip().split("\u241E")
                vector = np.array([float(el) for el in vector.split()])
                model["sentences"].append(sentence)
                model["vectors"].append(vector)
                model["labels"].append(label)
        return model


def make_save_path(full_path):
    if full_path[:4] == "data":
        full_path = os.path.join(os.path.abspath("."), full_path)
    model_path = '/'.join(full_path.split("/")[:-1])
    if not os.path.exists(model_path):
       os.makedirs(model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, help='method')
    parser.add_argument('--input_path', type=str, help='Location of input files')
    parser.add_argument('--output_path', type=str, help='Location of output files')
    parser.add_argument('--embedding_path', type=str, help='Location of embedding model')
    parser.add_argument('--is_weighted', type=str, help='Use weighted method or not')
    parser.add_argument('--train_corpus_path', type=str, help='Location of train corpus')
    parser.add_argument('--test_corpus_path', type=str, help='Location of test corpus')
    parser.add_argument('--embedding_name', type=str, help='embedding name')
    parser.add_argument('--embedding_corpus_path', type=str, help='embedding corpus path')
    parser.add_argument('--average', type=str, default="False", help='average or not')
    args = parser.parse_args()

    def str2bool(str):
        return str.lower() in ["true", "t"]

    if args.method == "train_word2vec":
        train_word2vec(args.input_path, args.output_path)
    elif args.method == "latent_semantic_analysis":
        latent_semantic_analysis(args.input_path, args.output_path)
    elif args.method == "cbow":
        model = CBoWModel(args.train_corpus_path, args.embedding_path,
                          args.output_path, args.embedding_corpus_path,
                          args.embedding_name, str2bool(args.is_weighted),
                          str2bool(args.average))
        model.evaluate(args.test_corpus_path)