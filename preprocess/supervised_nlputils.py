import sys, re, json
from khaiii import KhaiiiApi
from konlpy.tag import Okt, Komoran, Mecab, Hannanum


def tokenize(tokenizer_name, corpus_fname, output_fname, pos=False):
    if tokenizer_name == "komoran":
        tokenizer = Komoran()
    elif tokenizer_name == "okt":
        tokenizer = Okt()
    elif tokenizer_name == "mecab":
        tokenizer = Mecab()
    elif tokenizer_name == "hannanum":
        tokenizer = Hannanum()
    elif tokenizer_name == "khai":
        tokenizer = KhaiiiApi()
    else:
        tokenizer = Mecab()
    with open(corpus_fname, 'r', encoding='utf-8') as f1, \
            open(output_fname, 'w', encoding='utf-8') as f2:
        for line in f1:
            sentence = line.replace('\n', '').strip()
            if tokenizer_name == "khai":
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


def process_korsquad(corpus_fname, output_fname):
    with open(corpus_fname) as f1, \
        open(output_fname, 'w', encoding='utf-8') as f2:
        dataset_json = json.load(f1)
        dataset = dataset_json['data']
        for article in dataset:
            w_lines = []
            for paragraph in article['paragraphs']:
                w_lines.append(paragraph['context'])
                for qa in paragraph['qas']:
                    q_text = qa['question']
                    for a in qa['answers']:
                        a_text = a['text']
                        w_lines.append(q_text + " " + a_text)
            for line in w_lines:
                f2.writelines(line + "\n")


if __name__ == '__main__':
    tokenizer_name = sys.argv[1]
    in_f = sys.argv[2]
    out_f = sys.argv[3]
    if tokenizer_name == "korsquad":
        process_korsquad(in_f, out_f)
    else:
        tokenize(tokenizer_name, in_f, out_f)