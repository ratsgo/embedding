import sys
from gensim import corpora
from gensim.models import Doc2Vec, ldamulticore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def latent_semantic_analysis(corpus_fname, output_fname):
    corpus = [sent.replace('\n', '').strip() for sent in open(corpus_fname, 'r').readlines()]
    # construct tf-idf matrix
    vectorizer = TfidfVectorizer(
        min_df=10,
        ngram_range=(1, 1),
        lowercase=True,
        tokenizer=lambda x: x.split())
    input_matrix = vectorizer.fit_transform(corpus)
    # compute truncated SVD
    svd = TruncatedSVD(n_components=128)
    vecs = svd.fit_transform(input_matrix)
    with open(output_fname, 'w') as f:
        for doc_idx, vec in enumerate(vecs):
            str_vec = [str(el) for el in vec]
            f.writelines(str(doc_idx) + ' ' + ' '.join(str_vec) + "\n")


def doc2vec(corpus_fname, output_fname):
    # TODO: doc2vec용 코퍼스 찾고 작동하게 만들기
    corpus = [sent.replace('\n', '').strip().split(" ") for sent in open(corpus_fname, 'r').readlines()]
    model = Doc2Vec(corpus)
    model.save(output_fname)


def latent_dirichlet_allocation(corpus_fname, output_fname):
    raw_corpus = [sent.replace('\n', '').strip().split(" ") for sent in open(corpus_fname, 'r').readlines()]
    dictionary = corpora.Dictionary(raw_corpus)  # token : idx
    corpus = [dictionary.doc2bow(text) for text in raw_corpus]  # construct DTM, (token_id, freq)
    LDA = ldamulticore.LdaMulticore(corpus, id2word=dictionary,
                                    num_topics=100,
                                    minimum_probability=0.,
                                    workers=4)
    with open(output_fname + ".vecs", 'w') as f:
        for doc_idx, doc in enumerate(corpus):
            # TODO : topic dist가 모두 uniform하게 나오는 현상 해결
            doc_topic_distribution = [str(el[1]) for el in LDA[doc]]
            f.writelines(str(doc_idx) + ' ' + ' '.join(doc_topic_distribution) + "\n")
    LDA.save(output_fname + ".model")


if __name__ == '__main__':
    util_mode = sys.argv[1]
    in_f = sys.argv[2]
    out_f = sys.argv[3]
    if util_mode == "latent_semantic_analysis":
        latent_semantic_analysis(in_f, out_f)
    elif util_mode == "doc2vec":
        doc2vec(in_f, out_f)
    elif util_mode == "latent_dirichlet_allocation":
        latent_dirichlet_allocation(in_f, out_f)