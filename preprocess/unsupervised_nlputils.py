import sys, math
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer
from soynlp.normalizer import *
from soyspacing.countbase import CountSpace
from pytorch_pretrained_bert.tokenization import BertTokenizer


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


def compute_soy_word_score(corpus_fname, model_fname):
    sentences = [sent.replace('\n', '').strip() for sent in open(corpus_fname, 'r').readlines()]
    word_extractor = WordExtractor(min_frequency=100,
                                   min_cohesion_forward=0.05,
                                   min_right_branching_entropy=0.0
                                   )
    word_extractor.train(sentences)
    word_extractor.save(model_fname)


def soy_tokenize(corpus_fname, model_fname, output_fname):
    word_extractor = WordExtractor(min_frequency=100,
                                   min_cohesion_forward=0.05,
                                   min_right_branching_entropy=0.0
                                   )
    word_extractor.load(model_fname)
    scores = word_extractor.word_scores()
    # https://github.com/lovit/soynlp/blob/master/tutorials/wordextractor_lecture.ipynb
    # (1) 주어진 글자가 유기적으로 연결되어 함께 자주 나타나고,
    # (2) 그 단어의 우측에 다양한 조사, 어미, 혹은 다른 단어가 등장하여 단어의 우측의 branching entropy가 높다
    scores = {key:(scores[key].cohesion_forward * math.exp(scores[key].right_branching_entropy)) for key in scores.keys()}
    tokenizer = LTokenizer(scores=scores)
    with open(corpus_fname, 'r', encoding='utf-8') as f1, \
            open(output_fname, 'w', encoding='utf-8') as f2:
        for line in f1:
            sentence = line.replace('\n', '').strip()
            normalized_sent = emoticon_normalize(sentence, num_repeats=3)
            tokens = tokenizer.tokenize(normalized_sent)
            tokenized_sent = ' '.join(tokens)
            f2.writelines(tokenized_sent + '\n')


def process_sp_vocab(vocab_fname, output_fname):
    with open(vocab_fname, 'r', encoding='utf-8') as f1, \
            open(output_fname, 'w', encoding='utf-8') as f2:
        for line in f1:
            word = line.replace('\n', '').split('\t')[0].replace('▁', '##')
            if not word: continue
            f2.writelines(word + "\n")


def sentencepiece_tokenize(vocab_fname, corpus_fname, output_fname):
    tokenizer = BertTokenizer(vocab_file=vocab_fname, do_lower_case=False)
    with open(corpus_fname, 'r', encoding='utf-8') as f1, \
            open(output_fname, 'w', encoding='utf-8') as f2:
        for line in f1:
            sentence = line.replace('\n', '').strip()
            tokens = tokenizer.tokenize(sentence)
            tokenized_sent = ' '.join(tokens)
            f2.writelines(tokenized_sent + '\n')


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
    elif preprocess_mode == "compute_soy_word_score":
        in_f = sys.argv[2]
        model_f = sys.argv[3]
        compute_soy_word_score(in_f, model_f)
    elif preprocess_mode == "soy_tokenize":
        in_f = sys.argv[2]
        model_f = sys.argv[3]
        out_f = sys.argv[4]
        soy_tokenize(in_f, model_f, out_f)
    elif preprocess_mode == "process_sp_vocab":
        in_f = sys.argv[2]
        out_f = sys.argv[3]
        process_sp_vocab(in_f, out_f)
    elif preprocess_mode == "sentencepiece_tokenize":
        voca_f = sys.argv[2]
        in_f = sys.argv[3]
        out_f = sys.argv[4]
        sentencepiece_tokenize(voca_f, in_f, out_f)