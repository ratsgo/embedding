import sys, math, argparse
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer
from soynlp.normalizer import *
from soyspacing.countbase import CountSpace

sys.path.append('models')
from bert.tokenization import FullTokenizer, convert_to_unicode


def train_space_model(corpus_fname, model_fname):
    model = CountSpace()
    model.train(corpus_fname)
    model.save_model(model_fname, json_format=False)


def apply_space_correct(corpus_fname, model_fname, output_corpus_fname, with_label=False):
    model = CountSpace()
    model.load_model(model_fname, json_format=False)
    with open(corpus_fname, 'r', encoding='utf-8') as f1, \
        open(output_corpus_fname, 'w', encoding='utf-8') as f2:
        for sentence in f1:
            if with_label:
                sentence, label = sentence.strip().split("\u241E")
            else:
                sentence = sentence.strip()
                label = None
            if not sentence: continue
            sent_corrected, _ = model.correct(sentence)
            if with_label:
                f2.writelines(sent_corrected + "\u241E" + label + "\n")
            else:
                f2.writelines(sent_corrected + "\n")


def compute_soy_word_score(corpus_fname, model_fname):
    sentences = [sent.strip() for sent in open(corpus_fname, 'r').readlines()]
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
            if not word or word == "##": continue
            f2.writelines(word + "\n")


def sentencepiece_tokenize(vocab_fname, corpus_fname, output_fname):
    tokenizer = FullTokenizer(vocab_file=vocab_fname, do_lower_case=False)
    with open(corpus_fname, 'r', encoding='utf-8') as f1, \
            open(output_fname, 'w', encoding='utf-8') as f2:
        for line in f1:
            sentence = line.replace('\n', '').strip()
            tokens = tokenizer.tokenize(convert_to_unicode(sentence))
            tokenized_sent = ' '.join(tokens)
            f2.writelines(tokenized_sent + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess_mode', type=str, help='preprocess mode')
    parser.add_argument('--input_path', type=str, help='Location of input files')
    parser.add_argument('--output_path', type=str, help='Location of output files')
    parser.add_argument('--model_path', type=str, help='Location of model files')
    parser.add_argument('--vocab_path', type=str, help='Location of vocab files')
    parser.add_argument('--with_label', help='with label', type=str, default="False")
    args = parser.parse_args()

    if args.preprocess_mode == "train_space":
        train_space_model(args.input_path, args.model_path)
    elif args.preprocess_mode == "apply_space_correct":
        apply_space_correct(args.input_path, args.model_path, args.output_path, args.with_label.lower() == "true")
    elif args.preprocess_mode == "compute_soy_word_score":
        compute_soy_word_score(args.input_path, args.model_path)
    elif args.preprocess_mode == "soy_tokenize":
        soy_tokenize(args.input_path, args.model_path, args.output_path)
    elif args.preprocess_mode == "process_sp_vocab":
        process_sp_vocab(args.input_path, args.vocab_path)
    elif args.preprocess_mode == "sentencepiece_tokenize":
        sentencepiece_tokenize(args.vocab_path, args.input_path, args.output_path)