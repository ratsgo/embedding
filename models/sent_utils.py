import sys, os, argparse
from collections import Counter
from gensim import corpora
from gensim.models import Doc2Vec, ldamulticore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.models.doc2vec import TaggedDocument

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocess import get_tokenizer

sys.path.append('models')
from bilm import dump_weights as dump_elmo_weights


def latent_semantic_analysis(corpus_fname, output_fname, tokenizer_name="mecab"):
    make_save_path(output_fname)
    tokenizer = get_tokenizer(tokenizer_name)
    titles, raw_corpus, noun_corpus = [], [], []
    with open(corpus_fname, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                title, document = line.strip().split("\u241E")
                titles.append(title)
                raw_corpus.append(document)
                nouns = tokenizer.nouns(document)
                noun_corpus.append(' '.join(nouns))
            except:
                continue
    # construct tf-idf matrix
    vectorizer = TfidfVectorizer(
        min_df=1,
        ngram_range=(1, 1),
        lowercase=True,
        tokenizer=lambda x: x.split())
    input_matrix = vectorizer.fit_transform(noun_corpus)
    # compute truncated SVD
    svd = TruncatedSVD(n_components=100)
    vecs = svd.fit_transform(input_matrix)
    with open(output_fname, 'w') as f:
        for doc_idx, vec in enumerate(vecs):
            str_vec = [str(el) for el in vec]
            f.writelines(titles[doc_idx] + "\u241E" + raw_corpus[doc_idx] + '\u241E' + ' '.join(str_vec) + "\n")


class Doc2VecInput:

    def __init__(self, fname, tokenizer_name="mecab"):
        self.fname = fname
        self.tokenizer = get_tokenizer(tokenizer_name)

    def __iter__(self):
        with open(self.fname, encoding='utf-8') as f:
            for line in f:
                try:
                    sentence, movie_id = line.strip().split("\u241E")
                    tokens = self.tokenizer.morphs(sentence)
                    tagged_doc = TaggedDocument(words=tokens, tags=['MOVIE_%s' % movie_id])
                    yield tagged_doc
                except:
                    continue


def doc2vec(corpus_fname, output_fname):
    make_save_path(output_fname)
    corpus = Doc2VecInput(corpus_fname)
    model = Doc2Vec(corpus, vector_size=100)
    model.save(output_fname)


def latent_dirichlet_allocation(corpus_fname, output_fname, tokenizer_name="mecab"):
    make_save_path(output_fname)
    documents, tokenized_corpus = [], []
    tokenizer = get_tokenizer(tokenizer_name)
    with open(corpus_fname, 'r', encoding='utf-8') as f:
        for document in f:
            tokens = list(set(tokenizer.morphs(document.strip())))
            documents.append(document)
            tokenized_corpus.append(tokens)
    dictionary = corpora.Dictionary(tokenized_corpus)
    corpus = [dictionary.doc2bow(text) for text in tokenized_corpus]
    LDA = ldamulticore.LdaMulticore(corpus, id2word=dictionary,
                                    num_topics=30,
                                    minimum_probability=0.0,
                                    workers=4)
    # 특정 토픽의 확률이 0.5보다 클 경우에만 데이터를 리턴한다
    # 확률의 합은 1이기 때문에 해당 토픽이 해당 문서에서 확률값이 가장 큰 토픽이 된다
    all_topics = LDA.get_document_topics(corpus, minimum_probability=0.5, per_word_topics=False)
    with open(output_fname + ".results", 'w') as f:
        for doc_idx, topic in enumerate(all_topics):
            if len(topic) == 1:
                topic_id, prob = topic[0]
                f.writelines(documents[doc_idx].strip() + "\u241E" + ' '.join(tokenized_corpus[doc_idx]) + "\u241E" + str(topic_id) + "\u241E" + str(prob) + "\n")
    LDA.save(output_fname + ".model")


def construct_elmo_vocab(corpus_fname, output_fname):
    make_save_path(output_fname)
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


def make_save_path(full_path):
    model_path = '/'.join(full_path.split("/")[:-1])
    if not os.path.exists(model_path):
       os.makedirs(model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, help='method')
    parser.add_argument('--input_path', type=str, help='Location of input files')
    parser.add_argument('--output_path', type=str, help='Location of output files')
    args = parser.parse_args()

    if args.method == "latent_semantic_analysis":
        latent_semantic_analysis(args.input_path, args.output_path)
    elif args.method == "doc2vec":
        doc2vec(args.input_path, args.output_path)
    elif args.method == "latent_dirichlet_allocation":
        latent_dirichlet_allocation(args.input_path, args.output_path)
    elif args.method == "construct_elmo_vocab":
        construct_elmo_vocab(args.input_path, args.output_path)
    elif args.method == "dump_elmo_weights":
        dump_elmo_weights(args.input_path, args.output_path)