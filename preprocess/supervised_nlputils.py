import sys, re, argparse
from khaiii import KhaiiiApi
from konlpy.tag import Okt, Komoran, Mecab, Hannanum, Kkma


def get_tokenizer(tokenizer_name):
    if tokenizer_name == "komoran":
        tokenizer = Komoran()
    elif tokenizer_name == "okt":
        tokenizer = Okt()
    elif tokenizer_name == "mecab":
        tokenizer = Mecab()
    elif tokenizer_name == "hannanum":
        tokenizer = Hannanum()
    elif tokenizer_name == "kkma":
        tokenizer = Kkma()
    elif tokenizer_name == "khaiii":
        tokenizer = KhaiiiApi()
    else:
        tokenizer = Mecab()
    return tokenizer


def tokenize(tokenizer_name, corpus_fname, output_fname, pos=False):
    tokenizer = get_tokenizer(tokenizer_name)
    with open(corpus_fname, 'r', encoding='utf-8') as f1, \
            open(output_fname, 'w', encoding='utf-8') as f2:
        for line in f1:
            sentence = line.replace('\n', '').strip()
            if tokenizer_name == "khaiii":
                tokens = []
                for word in tokenizer.analyze(sentence):
                    if pos:
                        tokens.extend([str(m) for m in word.morphs])
                    else:
                        tokens.extend([str(m).split("/")[0] for m in word.morphs])
            else:
                if pos:
                    tokens = tokenizer.pos(sentence)
                    tokens = [morph + "/" + tag for morph, tag in tokens]
                else:
                    tokens = tokenizer.morphs(sentence)
            tokenized_sent = ' '.join(post_processing(tokens))
            f2.writelines(tokenized_sent + '\n')


def post_processing(tokens):
    results = []
    for token in tokens:
        # 숫자에 공백을 주어서 띄우기
        processed_token = [el for el in re.sub(r"(\d)", r" \1 ", token).split(" ") if len(el) > 0]
        results.extend(processed_token)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', type=str, help='tokenizer name')
    parser.add_argument('--input_path', type=str, help='Location of input files')
    parser.add_argument('--output_path', type=str, help='Location of output files')
    args = parser.parse_args()
    tokenize(args.tokenizer, args.input_path, args.output_path)