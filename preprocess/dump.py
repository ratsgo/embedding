import re, json, glob, argparse
from gensim.corpora import WikiCorpus, Dictionary
from gensim.utils import to_unicode

"""
Creates a corpus from Wikipedia dump file.
Inspired by:
https://www.kdnuggets.com/2017/11/building-wikipedia-text-corpus-nlp.html
"""
def make_corpus(in_f, out_f):
    """Convert Wikipedia xml dump file to text corpus"""
    output = open(out_f, 'w')
    wiki = WikiCorpus(in_f, tokenizer_func=tokenize, dictionary=Dictionary())
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
URL_PATTERN = re.compile("(ftp|http|https)?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", re.UNICODE)
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


def process_nsmc(corpus_path, output_fname, process_json=True, with_label=True):
    if process_json:
        file_paths = glob.glob(corpus_path + "/*")
        with open(output_fname, 'w', encoding='utf-8') as f:
            for path in file_paths:
                contents = json.load(open(path))
                for content in contents:
                    sentence = content['review'].strip()
                    if len(sentence) > 0:
                        f.writelines(sentence + "\u241E" + content['movie_id'] + "\n")
    else:
        with open(corpus_path, 'r', encoding='utf-8') as f1, \
                open(output_fname, 'w', encoding='utf-8') as f2:
            next(f1)  # skip head line
            for line in f1:
                _, sentence, label = line.strip().split('\t')
                if not sentence: continue
                if with_label:
                    f2.writelines(sentence + "\u241E" + label + "\n")
                else:
                    f2.writelines(sentence + "\n")


def process_korQuAD(corpus_fname, output_fname):
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


def process_documents(corpus_fname, output_fname):
    with open(corpus_fname) as f1, open(output_fname, 'w', encoding='utf-8') as f2:
        for line in f1:
            sentences = re.split("(?<=[.!?])\s+", line.strip())
            for sentence in sentences:
                f2.writelines(sentence + "\n")
            f2.writelines("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess_mode', type=str, help='preprocess mode')
    parser.add_argument('--input_path', type=str, help='Location of input files')
    parser.add_argument('--output_path', type=str, help='Location of output files')
    parser.add_argument('--with_label', help='with label', type=str, default="False")
    args = parser.parse_args()

    if args.preprocess_mode == "wiki":
        make_corpus(args.input_path, args.output_path)
    elif "nsmc" in args.preprocess_mode:
        process_nsmc(args.input_path, args.output_path, "json" in args.preprocess_mode, args.with_label.lower() == "true")
    elif args.preprocess_mode == "korquad":
        process_korQuAD(args.input_path, args.output_path)
    elif args.preprocess_mode == "process-documents":
        process_documents(args.input_path, args.output_path)
