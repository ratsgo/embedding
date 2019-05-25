import sys, os
import tensorflow as tf
from collections import Counter
from gensim import corpora
from gensim.models import Doc2Vec, ldamulticore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.models.doc2vec import TaggedDocument

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocess import get_tokenizer

sys.path.append('models')
from bilm import Batcher, BidirectionalLanguageModel, weight_layers
from bilm import dump_weights as dump_elmo_weights


def latent_semantic_analysis(corpus_fname, output_fname):
    tokenizer = get_tokenizer("mecab")
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


def extract_elmo_embeddings(sentence_fname, model_fname,
                            options_fname,
                            vocab_fname, output_fname,
                            tune=False,
                            max_char_length=30):
    # Create a Batcher to map text to character ids.
    batcher = Batcher(vocab_fname, max_char_length)
    # Input placeholders to the biLM.
    character_ids = tf.placeholder('int32', shape=(None, None, max_char_length))
    # Build the biLM graph.
    bilm = BidirectionalLanguageModel(options_fname, model_fname)
    # Get ops to compute the LM embeddings.
    embeddings_op = bilm(character_ids)
    # Get ELMo embeddings
    # 정석대로라면 특정 task 수행을 위한 튜닝 과정에서 구축되는,
    # BiLM의 모든 벡터들의 weighted sum이 ELMo embeddings임
    # 하지만 컴퓨팅 리소스가 부족하고 BiLM 자체의 품질을 확인하고 싶을 때
    # BiLM의 벡터들을 그대로 뽑아 본다
    if tune:
        elmo_embeddings = weight_layers('elmo_embeddings', embeddings_op, l2_coef=0.0)
    # load corpus
    # 학습 데이터와 같은 토크나이저를 사용한 tokenized corpus여야 한다
    corpus = [line.strip().split(" ") for line in open(sentence_fname, 'r')]
    # extract ELMo embeddings
    with tf.Session() as sess:
        # It is necessary to initialize variables once before running inference.
        sess.run(tf.global_variables_initializer())
        with open(output_fname, 'w') as f:
            for tokenized_sent in corpus:
                # Create batches of data.
                ids = batcher.batch_sentences([tokenized_sent])
                # Compute ELMo representations
                if tune:
                    vector = sess.run(elmo_embeddings['weighted_op'], feed_dict={character_ids: ids})
                else:
                    vector = sess.run(embeddings_op, feed_dict={character_ids: ids})
                str_vector = [str(el) for el in vector]
                f.writelines(' '.join(str_vector) + "\n")


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
    elif util_mode == "dump_elmo_weights":
        dump_elmo_weights(in_f, out_f)
    elif util_mode == "extract_elmo_embeddings":
        extract_elmo_embeddings()