import sys
from gensim.models import Word2Vec


def train_word2vec(corpus_fname, model_fname):
    corpus = [sent.replace('\n', '').strip().split(" ") for sent in open(corpus_fname, 'r').readlines()]
    model = Word2Vec(corpus, size=128, workers=4, sg=1)
    model.save(fname=model_fname)


if __name__ == '__main__':
    util_mode = sys.argv[1]
    if util_mode == "train_word2vec":
        in_f = sys.argv[2]
        out_f = sys.argv[3]
        train_word2vec(in_f, out_f)