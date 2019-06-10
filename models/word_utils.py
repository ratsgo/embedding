import argparse, os
from gensim.models import Word2Vec
from sklearn.decomposition import TruncatedSVD
from soynlp.word import pmi
from soynlp.vectorizer import sent_to_word_contexts_matrix


def train_word2vec(corpus_fname, model_fname):
    make_save_path(model_fname)
    corpus = [sent.replace('\n', '').strip().split(" ") for sent in open(corpus_fname, 'r').readlines()]
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
    pmi_matrix, _, _ = pmi(input_matrix, min_pmi=0, alpha=0)
    # compute truncated SVD
    pmi_svd = TruncatedSVD(n_components=100)
    pmi_vecs = pmi_svd.fit_transform(input_matrix)
    with open(output_fname + "-pmi.vecs", 'w') as f2:
        for word, vec in zip(idx2vocab, pmi_vecs):
            str_vec = [str(el) for el in vec]
            f2.writelines(word + ' ' + ' '.join(str_vec) + "\n")


def construct_weighted_embedding(corpus_fname, embedding_fname, output_fname):
    # compute word frequency
    # compute weighted vectors
    return None


def make_save_path(full_path):
    model_path = '/'.join(full_path.split("/")[:-1])
    if not os.path.exists(model_path):
       os.mkdir(model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, help='method')
    parser.add_argument('--input_path', type=str, help='Location of input files')
    parser.add_argument('--output_path', type=str, help='Location of output files')
    args = parser.parse_args()

    if args.method == "train_word2vec":
        train_word2vec(args.input_path, args.output_path)
    elif args.method == "latent_semantic_analysis":
        args.latent_semantic_analysis(args.input_path, args.output_path)
    elif args.method == "construct_weighted_embedding":
        construct_weighted_embedding(args.input_path, args.output_path)