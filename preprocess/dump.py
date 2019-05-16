import sys, re, json, glob
from gensim.corpora import WikiCorpus
from gensim.utils import to_unicode
from preprocess import get_tokenizer, post_processing

"""
Creates a corpus from Wikipedia dump file.
Inspired by:
https://www.kdnuggets.com/2017/11/building-wikipedia-text-corpus-nlp.html
"""
def make_corpus(in_f, out_f):
    """Convert Wikipedia xml dump file to text corpus"""
    output = open(out_f, 'w')
    wiki = WikiCorpus(in_f, tokenizer_func=tokenize)
    i = 0
    for text in wiki.get_texts():
        output.write(bytes(' '.join(text), 'utf-8').decode('utf-8') + '\n')
        i = i + 1
        if (i % 10000 == 0):
            print('Processed ' + str(i) + ' articles')
    output.close()
    print('Processing complete!')


WIKI_REMOVE_CHARS = re.compile("'+|(=+.{2,30}=+)|__TOC__|(ファイル:).+|:(en|de|it|fr|es|kr|zh|no|fi):|\n", re.UNICODE)
WIKI_SPACE_CHARS = re.compile("(\\s|゙|゚|　)+", re.UNICODE)
EMAIL_PATTERN = re.compile("(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)", re.UNICODE)
URL_PATTERN = re.compile("((ftp|http|https):\/\/)?(www.)?(?!.*(ftp|http|https|www.))[a-zA-Z0-9_-]+(\.[a-zA-Z]+)+((\/)[\w#]+)*(\/\w+\?[a-zA-Z0-9_]+=\w+(&[a-zA-Z0-9_]+=\w+)*)?$", re.UNICODE)
WIKI_REMOVE_TOKEN_CHARS = re.compile("(\\*$|:$|^파일:.+|^;)", re.UNICODE)
MULTIPLE_SPACES = re.compile(' +', re.UNICODE)


def tokenize(content, token_min_len=2, token_max_len=100, lower=True):
    content = re.sub(EMAIL_PATTERN, ' ', content)  # remove email pattern
    content = re.sub(URL_PATTERN, ' ', content) # remove url pattern
    content = re.sub(WIKI_REMOVE_CHARS, ' ', content)  # remove unnecessary chars
    content = re.sub(WIKI_SPACE_CHARS, ' ', content)
    content = re.sub(MULTIPLE_SPACES, ' ', content)
    tokens = content.replace(", )", "").split(" ")
    result = []
    for token in tokens:
        if not token.startswith('_'):
            token_candidate = to_unicode(re.sub(WIKI_REMOVE_TOKEN_CHARS, '', token))
        else:
            token_candidate = ""
        if len(token_candidate) > 0:
            result.append(token_candidate)
    return result


def process_nsmc(corpus_path, output_fname, process_json=True):
    tokenizer = get_tokenizer("mecab")
    if process_json:
        file_paths = glob.glob(corpus_path + "/*")
        with open(output_fname, 'w', encoding='utf-8') as f:
            for path in file_paths:
                contents = json.load(open(path))
                for content in contents:
                    tokens = tokenizer.morphs(content['review'])
                    tokenized_sent = ' '.join(post_processing(tokens))
                    f.writelines(tokenized_sent + "\u241E" + content['movie_id'] + "\n")
    else:
        with open(corpus_path, 'r', encoding='utf-8') as f1, \
                open(output_fname, 'w', encoding='utf-8') as f2:
            next(f1)  # skip head line
            for line in f1:
                sentence = line.replace('\n', '').split('\t')[1]
                if not sentence: continue
                f2.writelines(sentence + "\n")


def process_korsquad(corpus_fname, output_fname):
    with open(corpus_fname) as f1, open(output_fname, 'w', encoding='utf-8') as f2:
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
    preprocess_mode = sys.argv[1]
    in_f = sys.argv[2]
    out_f = sys.argv[3]
    if preprocess_mode == "wiki":
        make_corpus(in_f, out_f)
    elif "nsmc" in preprocess_mode:
        process_nsmc(in_f, out_f, "json" in preprocess_mode)
    elif preprocess_mode == "korsquad":
        process_korsquad(in_f, out_f)
