import sys
import numpy as np
import scipy.stats as st
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize
from preprocess import get_tokenizer

sys.path.append('models')
from visualize_utils import visualize_words, visualize_between_words


class WordEmbeddingEval:

    def __init__(self, vecs_fname, method="word2vec", dim=100, tokenize=True):
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
        return sorted(zip(self.words, scores), key=lambda x: x[1], reverse=True)[1:topn+1]

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
        print(spearman, pearson, missed)

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
                        print(words[0] + " + " + words[1] + " - " + words[2])
                        print("correct answer:", words[3])
                        print("predicted answers:", word_with_scores)
                        print("")
                    similar_words = [el[0] for el in word_with_scores]
                    if words[-1] in similar_words:
                        correct += 1
                except:
                    missed += 1
                total += 1
        print(correct, total, missed)

    def get_analogy_vector(self, words):
        if len(words) == 3:
            token_1 = self.get_sentence_vector(words[0])
            token_2 = self.get_sentence_vector(words[1])
            token_3 = self.get_sentence_vector(words[2])
            result = token_1 - token_2 + token_3
        else:
            result = np.zeros(self.dim)
        return result

    # token vector들을 lookup한 뒤 평균을 취한다
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
            result = np.mean(token_vecs, axis=0)
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
        for line in open(words_fname, 'r'):
            if not line.startswith("#"):
                for word in line.strip().split(" "):
                    words.add(word)
        vecs = np.array([self.get_sentence_vector(word) for word in words])
        visualize_words(words, vecs, palette)

    def visualize_between_words(self, words_fname, palette="Viridis256"):
        words = set()
        for line in open(words_fname, 'r'):
            if not line.startswith("#"):
                for word in line.strip().split(" "):
                    words.add(word)
        vecs = [self.get_sentence_vector(word) for word in words]
        visualize_between_words(words, vecs, palette)



model = WordEmbeddingEval("data/word2vec.vecs", "word2vec", dim=100, tokenize=True)
model.word_sim_test("data/kor_ws353.csv") # 0.5770993871014621 0.5956751142850295 0
model.word_analogy_test("data/kor_analogy_semantic.txt") # 158 420 0
model.word_analogy_test("data/kor_analogy_syntactic.txt")
model.most_similar("문재인") # [('박근혜', 0.9221434), ('이명박', 0.9151239), ('노무현', 0.90462077), ('김대중', 0.8763489), ('노태우', 0.8345808), ('김영삼', 0.8336141), ('이회창', 0.8229812), ('박원순', 0.81067884), ('전두환', 0.8074833), ('안철수', 0.8040389)]
model.visualize_words("data/kor_analogy_semantic.txt", palette="Viridis256")
model.visualize_words("data/kor_analogy_syntactic.txt", palette="Greys256")
model.visualize_between_words("data/kor_analogy_semantic.txt", palette="Greys256")

model = WordEmbeddingEval("data/glove.vecs.txt", "glove", dim=100, tokenize=True)
model.word_sim_test("data/kor_ws353.csv") # 0.49029953452220065 0.5383746018370396 0
model.word_analogy_test("data/kor_analogy_semantic.txt") # 110 420 0
model.most_similar("문재인") # ('이명박', 0.845631133898592), ('박근혜', 0.8174952332797571), ('노무현', 0.8042187984352386), ('김대중', 0.7265464328203921), ('대통령', 0.7200989781982694), ('대선', 0.6938117143292233), ('문재', 0.6911781464368917), ('김영삼', 0.6797721738977291), ('이회창', 0.6789389835604196), ('청와대', 0.6771037890591527)]
model.visualize_words("data/kor_analogy_semantic.txt", palette="Inferno256")
model.visualize_words("data/kor_analogy_syntactic.txt", palette="Magma256")

model = WordEmbeddingEval("data/fasttext.vecs.vec", "fasttext", dim=100, tokenize=True)
model.word_sim_test("data/kor_ws353.csv") # 0.636179558597476 0.6386177571595193 0
model.word_analogy_test("data/kor_analogy_semantic.txt") # 81 420 0
model.most_similar("문재인") # [('박근혜', 0.9239191680881065), ('이명박', 0.9129016338164864), ('노무현', 0.8974644850690527), ('문재', 0.8477265842739639), ('노태우', 0.8271708135634908), ('김대중', 0.8241289233466147), ('청와대', 0.808890823843111), ('이회창', 0.8088008847778916), ('박원순', 0.8075874801817806), ('홍준표', 0.7973925010954298)]
model.visualize_words("data/kor_analogy_semantic.txt", palette="Plasma256")
model.visualize_words("data/kor_analogy_syntactic.txt", palette="Cividis256")

model = WordEmbeddingEval("data/swivel.vecs/row_embedding.tsv", "swivel", dim=100, tokenize=True)
model.word_sim_test("data/kor_ws353.csv") # 0.549541215508716 0.5727286333920304 0
model.word_analogy_test("data/kor_analogy_semantic.txt") # 92 420 0
model.most_similar("문재인") # [('이명박', 0.7587751892657401), ('박근혜', 0.7347895426965483), ('노무현', 0.720725618392337), ('청와대', 0.7050805661577202), ('대선', 0.7016249703619943), ('홍준표', 0.6821123111055576), ('안희정', 0.6746228614203431), ('이회창', 0.6746018345903241), ('한나라당', 0.6669747023554933), ('대통령', 0.66614570376648)]
model.visualize_words("data/kor_analogy_semantic.txt")
model.visualize_words("data/kor_analogy_syntactic.txt")

# TODO : word2vec-lsa vs lsa-pmi 차이 검증
model = WordEmbeddingEval("data/word2vec-lsa.vecs", "word2vec", dim=100, tokenize=True)
model.word_sim_test("data/kor_ws353.csv") # 0.44151727940351265 0.4246878668643376 0
model.word_analogy_test("data/kor_analogy_semantic.txt") # 94 420 0
model.most_similar("문재인") # [('청와대', 0.828278), ('김대중', 0.7862822), ('당선자', 0.78047204), ('이정희', 0.7796035), ('홍준표', 0.77874863), ('박원순', 0.7736834), ('야권', 0.77013516), ('유시민', 0.76993674), ('반기문', 0.76893556), ('원내대표', 0.76648307)]
model.visualize_words("data/kor_analogy_semantic.txt")
model.visualize_words("data/kor_analogy_syntactic.txt")

model = WordEmbeddingEval("data/lsa-pmi.vecs", "lsa-pmi", dim=100, tokenize=True)
model.word_sim_test("data/kor_ws353.csv") # 0.1824178080678877 0.16985325941114984 0
model.word_analogy_test("data/kor_analogy_semantic.txt") # 34 420 0
model.most_similar("문재인") # [('이명박', 0.9668353235506264), ('김대중', 0.965416730672013), ('이승만', 0.9642945139418547), ('노무현', 0.962421106788593), ('전두환', 0.9601622267707821), ('오딘', 0.9509565666301292), ('김영삼', 0.9489033288641142), ('이회창', 0.9479594521397682), ('나폴레옹', 0.9445780885897337), ('선장', 0.9415429819696624)]
model.visualize_words("data/kor_analogy_semantic.txt")
model.visualize_words("data/kor_analogy_syntactic.txt")

model = WordEmbeddingEval("data/lsa-cooc.vecs", "lsa-cooc", dim=100, tokenize=True)
model.word_sim_test("data/kor_ws353.csv") # 0.1824464486427594 0.16992081424024436 0
model.word_analogy_test("data/kor_analogy_semantic.txt") # 34 420 0
model.most_similar("문재인") # [('이명박', 0.9668487901512071), ('김대중', 0.9653821384884316), ('이승만', 0.9643050967698661), ('노무현', 0.9623744447003055), ('전두환', 0.9601431274381929), ('오딘', 0.9509792551908478), ('김영삼', 0.9489595747780442), ('이회창', 0.9478132961857701), ('나폴레옹', 0.9445455485081845), ('김연경', 0.9414856648101293)]
model.visualize_words("data/kor_analogy_semantic.txt")
model.visualize_words("data/kor_analogy_syntactic.txt")