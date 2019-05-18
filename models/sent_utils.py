import sys
from collections import Counter
from gensim import corpora
from gensim.models import Doc2Vec, ldamulticore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.models.doc2vec import TaggedDocument


def latent_semantic_analysis(corpus_fname, output_fname):
    corpus = []
    movie_ids = []
    with open(corpus_fname, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                _, tokens, movie_id = line.replace('\n', '').strip().split("\u241E")
                corpus.append(tokens)
                movie_ids.append(movie_id)
            except:
                continue
    # construct tf-idf matrix
    vectorizer = TfidfVectorizer(
        min_df=10,
        ngram_range=(1, 1),
        lowercase=True,
        tokenizer=lambda x: x.split())
    input_matrix = vectorizer.fit_transform(corpus)
    # compute truncated SVD
    svd = TruncatedSVD(n_components=100)
    vecs = svd.fit_transform(input_matrix)
    with open(output_fname, 'w') as f:
        for doc_idx, vec in enumerate(vecs):
            str_vec = [str(el) for el in vec]
            f.writelines(str(doc_idx) + ' ' + str(movie_ids[doc_idx]) + ' '.join(str_vec) + "\n")


class Doc2VecInput:

    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        with open(self.fname, encoding='utf-8') as f:
            for line in f:
                try:
                    _, tokens, movie_id = line.replace('\n', '').strip().split("\u241E")
                    tagged_doc = TaggedDocument(words=tokens.split(), tags=['MOVIE_%s' % movie_id])
                    yield tagged_doc
                except:
                    continue


def doc2vec(corpus_fname, output_fname):
    corpus = Doc2VecInput(corpus_fname)
    model = Doc2Vec(corpus, vector_size=100)
    model.save(output_fname)


def latent_dirichlet_allocation(corpus_fname, output_fname):
    raw_corpus = []
    movie_ids = []
    with open(corpus_fname, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                _, tokens, movie_id = line.replace('\n', '').strip().split("\u241E")
                raw_corpus.append(tokens.split(" "))
                movie_ids.append(movie_id)
            except:
                continue
    dictionary = corpora.Dictionary(raw_corpus)  # token : idx
    corpus = [dictionary.doc2bow(text) for text in raw_corpus]  # construct DTM, (token_id, freq)
    LDA = ldamulticore.LdaMulticore(corpus, id2word=dictionary,
                                    num_topics=30,
                                    minimum_probability=0.0,
                                    workers=4)
    # top 20 words for each topic
    # top_words = [(topic_num, [word for word, _ in LDA.show_topic(topic_num, topn=20)]) for topic_num in
    #               range(LDA.num_topics)]
    # get topic distribution of a document
    all_topics = LDA.get_document_topics(corpus, per_word_topics=False)
    with open(output_fname + ".vecs", 'w') as f:
        for doc_idx, doc_dist in enumerate(all_topics):
            # TODO : topic dist가 모두 uniform하게 나오는 현상 해결
            doc_topic_distribution = [str(el[1]) for el in doc_dist]
            f.writelines(str(doc_idx) + ' ' + str(movie_ids[doc_idx]) + ' ' + ' '.join(doc_topic_distribution) + "\n")
    LDA.save(output_fname + ".model")


def construct_elmo_vocab(corpus_fname, output_fname):
    count = Counter()
    with open(corpus_fname, 'r', encoding='utf-8') as f1:
        for sentence in f1:
            tokens = sentence.replace('\n', '').strip().split(" ")
            for token in tokens:
                count[token] += 1
    with open(output_fname, 'w', encoding='utf-8') as f2:
        f2.writelines("</S>\n")
        f2.writelines("<S>\n")
        f2.writelines("<UNK>\n")
        for word, _ in count.most_common(100000):
            f2.writelines(word + "\n")


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
    elif util_mode == "construct_elmo_vocab":
        construct_elmo_vocab(in_f, out_f)