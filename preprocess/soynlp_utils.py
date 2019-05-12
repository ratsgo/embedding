import sys, pickle
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer
from soynlp.normalizer import *
from soyspacing.countbase import CountSpace
from soynlp.noun import LRNounExtractor_v2


def train_space_model(corpus_fname, model_fname):
    model = CountSpace()
    model.train(corpus_fname)
    model.save_model(model_fname, json_format=False)


def apply_space_correct(corpus_fname, model_fname, output_corpus_fname):
    model = CountSpace()
    model.load_model(model_fname, json_format=False)
    with open(corpus_fname, 'r', encoding='utf-8') as f1, \
        open(output_corpus_fname, 'w', encoding='utf-8') as f2:
        for sentence in f1:
            sentence = sentence.replace('\n', '').strip()
            if not sentence: continue
            sent_corrected, _ = model.correct(sentence)
            f2.writelines(sent_corrected + "\n")


def process_nvc(corpus_fname, output_fname):
    with open(corpus_fname, 'r', encoding='utf-8') as f1, \
            open(output_fname, 'w', encoding='utf-8') as f2:
        next(f1) # skip head line
        for line in f1:
            sentence = line.replace('\n', '').split('\t')[1]
            if not sentence: continue
            f2.writelines(sentence + "\n")


def compute_word_score(corpus_fname, model_fname):
    sentences = [sent.replace('\n', '').strip() for sent in open(corpus_fname, 'r').readlines()]
    word_extractor = WordExtractor(min_frequency=100,
                                   min_cohesion_forward=0.05,
                                   min_right_branching_entropy=0.0
                                   )
    word_extractor.train(sentences)
    word_extractor.save(model_fname)


def tokenize(corpus_fname, model_fname, output_fname):
    word_extractor = WordExtractor(min_frequency=100,
                                   min_cohesion_forward=0.05,
                                   min_right_branching_entropy=0.0
                                   )
    word_extractor.load(model_fname)
    scores = word_extractor.word_scores()
    tokenizer = LTokenizer(scores=scores)
    with open(corpus_fname, 'r', encoding='utf-8') as f1, \
            open(output_fname, 'w', encoding='utf-8') as f2:
        for line in f1:
            sentence = line.replace('\n', '').strip()
            normalized_sent = emoticon_normalize(sentence, num_repeats=3)
            tokenized_sent = tokenizer.tokenize(normalized_sent)
            f2.writelines(tokenized_sent + '\n')


def train_noun(corpus_fname, model_fname):
    sentences = [sent.replace('\n', '').strip() for sent in open(corpus_fname, 'r').readlines()]
    noun_extractor = LRNounExtractor_v2(verbose=True)
    nouns = noun_extractor.train_extract(sentences)  # {str:namedtuple} 형식
    noun_extractor.predict("우리나라")
    with open(model_fname, 'wb') as f:
        pickle.dump(nouns, f)



if __name__ == '__main__':
    preprocess_mode = sys.argv[1]
    if preprocess_mode == "train_space":
        in_f = sys.argv[2]
        out_f = sys.argv[3]
        train_space_model(in_f, out_f)
    elif preprocess_mode == "apply_space_correct":
        in_f = sys.argv[2]
        model_f = sys.argv[3]
        out_f = sys.argv[4]
        apply_space_correct(in_f, model_f, out_f)
    elif preprocess_mode == "process_nvc":
        in_f = sys.argv[2]
        out_f = sys.argv[3]
        process_nvc(in_f, out_f)
    elif preprocess_mode == "compute_word_score":
        in_f = sys.argv[2]
        model_f = sys.argv[3]
        compute_word_score(in_f, model_f)
    elif preprocess_mode == "tokenize":
        in_f = sys.argv[2]
        model_f = sys.argv[3]
        out_f = sys.argv[4]
        tokenize(in_f, model_f, out_f)
    elif preprocess_mode == "train_noun":
        in_f = sys.argv[2]
        model_f = sys.argv[3]
        train_noun(in_f, model_f)