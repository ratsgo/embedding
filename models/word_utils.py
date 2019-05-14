import sys
from gensim.models import Word2Vec
from sklearn.decomposition import TruncatedSVD
from soynlp.word import pmi
from soynlp.vectorizer import sent_to_word_contexts_matrix


def train_word2vec(corpus_fname, model_fname):
    corpus = [sent.replace('\n', '').strip().split(" ") for sent in open(corpus_fname, 'r').readlines()]
    model = Word2Vec(corpus, size=128, workers=4, sg=1)
    model.save(model_fname)


def latent_semantic_analysis(corpus_fname, output_fname, mode):
    corpus = [sent.replace('\n', '').strip() for sent in open(corpus_fname, 'r').readlines()]
    # construct co-occurrence matrix (=word_context)
    # dynamic weight if True. co-occurrence weight = [1, (w-1)/w, (w-2)/w, ... 1/w]
    input_matrix, idx2vocab = sent_to_word_contexts_matrix(
        corpus,
        windows=3,
        min_tf=10,
        dynamic_weight=True,
        verbose=True)
    if mode == 'pmi':
        # Shift PPMI at k=0, (equal PPMI)
        # return
        # pmi(word, contexts)
        # px: Probability of rows(items)
        # py: Probability of columns(features)
        input_matrix, _, _ = pmi(input_matrix, min_pmi=0, alpha=0)
    # compute truncated SVD
    svd = TruncatedSVD(n_components=128)
    vecs = svd.fit_transform(input_matrix)
    with open(output_fname, 'w') as f:
        for word, vec in zip(idx2vocab, vecs):
            str_vec = [str(el) for el in vec]
            f.writelines(word + ' ' + ' '.join(str_vec) + "\n")


def construct_weighted_embedding(corpus_fname, embedding_fname, output_fname):
    # compute word frequency
    # compute weighted vectors
    return None


if __name__ == '__main__':
    util_mode = sys.argv[1]
    if util_mode == "train_word2vec":
        in_f = sys.argv[2]
        out_f = sys.argv[3]
        train_word2vec(in_f, out_f)
    elif util_mode == "latent_semantic_analysis":
        in_f = sys.argv[2]
        out_f = sys.argv[3]
        analysis_mode = sys.argv[4]
        latent_semantic_analysis(in_f, out_f, analysis_mode)