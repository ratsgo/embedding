import sys
sys.path.append('models')

import tensorflow as tf
from bert.modeling import BertModel, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.manifold import TSNE

from bokeh.io import show
from bokeh.plotting import figure
from bokeh.models import Plot, Range1d, MultiLine, Circle, HoverTool, TapTool, BoxSelectTool, LinearColorMapper, ColumnDataSource, LabelSet, SaveTool
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.palettes import Spectral8



class BERTEmbeddingEval:

    def __init__(self, model_fname="data/bert",
                 bertconfig_fname="data/bert/multi_cased_L-12_H-768_A-12/bert_config.json",
                 vocab_fname="data/bert/multi_cased_L-12_H-768_A-12/vocab.txt",
                 max_seq_length=32, dim=768):

        config = BertConfig.from_json_file(bertconfig_fname)
        self.tokenizer = BertTokenizer(vocab_file=vocab_fname, do_lower_case=False)
        self.max_seq_length = max_seq_length
        self.dim = dim
        self.input_ids = tf.placeholder(tf.int32, [1, max_seq_length], name='inputs_ids')
        self.input_mask = tf.placeholder(tf.int32, [1, max_seq_length], name='input_mask')
        self.segment_ids = tf.placeholder(tf.int32, [1, max_seq_length], name='segment_ids')
        self.model = BertModel(config=config,
                               is_training=False,
                               input_ids=self.input_ids,
                               input_mask=self.input_mask,
                               token_type_ids=self.segment_ids)
        saver = tf.train.Saver(tf.global_variables())
        self.sess = tf.Session()
        checkpoint_path = tf.train.latest_checkpoint(model_fname)
        saver.restore(self.sess, checkpoint_path)

    """
    sentence를 입력하면 토크나이즈 결과와 token 벡터 시퀀스를 반환한다
        - shape :[[# of tokens], [batch size, max seq length, dimension]]
    """
    def get_sentence_vector(self, sentence):
        tokens = self.tokenize(sentence)
        model_input = self.make_input(tokens)
        return [tokens, self.sess.run(self.model.get_sequence_output()[0], model_input)[:len(tokens)]]

    """
    sentence를 입력하면 토크나이즈 결과와 self-attention score matrix를 반환한다
        - shape :[[# of tokens], [batch size, # of tokens, # of tokens]]
    """
    def get_self_attention_score(self, sentence):
        tokens = self.tokenize(sentence)
        model_input = self.make_input(tokens)
        # raw_score : shape=[# of layers, batch_size, num_attention_heads, max_seq_length, max_seq_length]
        raw_score = self.sess.run(self.model.attn_probs_for_visualization_list, model_input)
        # 마지막 레이어를 취한 뒤, attention head 기준(axis=0)으로 sum
        scores = np.sum(raw_score[-1][0], axis=0)
        # scores matrix에서 토큰 개수만큼 취함
        scores = scores[:len(tokens), :len(tokens)]
        return [tokens, scores]

    def tokenize(self, sentence):
        return self.tokenizer.tokenize(sentence)

    def make_input(self, tokens):
        tokens = tokens[:(self.max_seq_length - 2)]
        token_sequence = ["[CLS]"] + tokens + ["[SEP]"]
        segment = [0] * len(token_sequence)
        sequence = self.tokenizer.convert_tokens_to_ids(token_sequence)
        current_length = len(sequence)
        padding_length = self.max_seq_length - current_length
        input_feed = {
            self.input_ids: np.array([sequence + [0] * padding_length]),
            self.segment_ids: np.array([segment + [0] * padding_length]),
            self.input_mask: np.array([[1] * current_length + [0] * padding_length])
        }
        return input_feed

    """
    Visualize Multiple Sentences (2d vector space)
    Inspired by:
    https://github.com/hengluchang/visualizing_contextual_vectors/blob/master/elmo_vis.py
    """
    def visualize_sentences(self, interest_word, sentences, palette="Viridis256"):
        tokenized_sentences = []
        vecs = np.zeros((1, self.dim))
        for sentence in sentences:
            tokens, vec = self.get_sentence_vector(sentence)
            tokenized_sentences.append(tokens)
            vecs = np.concatenate([vecs, vec], axis=0)
        # process sentences
        token_list, processed_sentences = [], []
        for tokens in tokenized_sentences:
            token_list.extend(tokens)
            sentence = []
            for token in tokens:
                processed_token = token.replace("##", "")
                if token == interest_word:
                    processed_token = "\"" + processed_token + "\""
                sentence.append(processed_token)
            processed_sentences.append(' '.join(sentence))
        # dimension reduction
        tsne = TSNE(n_components=2)
        tsne_results = tsne.fit_transform(vecs[1:])
        # only plot the word representation of interest
        interest_vecs, idx = np.zeros((len(sentences), 2)), 0
        for word, vec in zip(token_list, tsne_results):
            if word == interest_word:
                interest_vecs[idx] = vec
                idx += 1
        df = pd.DataFrame(columns=['x', 'y', 'annotation'])
        df['x'], df['y'], df['annotation'] = interest_vecs[:, 0], interest_vecs[:, 1], processed_sentences
        source = ColumnDataSource(ColumnDataSource.from_df(df))
        labels = LabelSet(x="x", y="y", text="annotation", y_offset=8,
                          text_font_size="8pt", text_color="#555555",
                          source=source, text_align='center')
        color_mapper = LinearColorMapper(palette=palette, low=min(tsne_results[:, 1]), high=max(tsne_results[:, 1]))
        plot = figure(plot_width=900, plot_height=900)
        plot.scatter("x", "y", size=12, source=source, color={'field': 'y', 'transform': color_mapper}, line_color=None,
                     fill_alpha=0.8)
        plot.add_layout(labels)
        show(plot, notebook_handle=True)

    def visualize_self_attention_scores(self, sentence, palette="Viridis256"):
        tokens, scores = self.get_self_attention_score(sentence)
        mean_prob = np.mean(scores)
        weighted_edges = []
        for idx_1, token_prob_dist_1 in enumerate(scores):
            for idx_2, el in enumerate(token_prob_dist_1):
                if idx_1 == idx_2 or el < mean_prob:
                    weighted_edges.append((tokens[idx_1], tokens[idx_2], 0))
                else:
                    weighted_edges.append((tokens[idx_1], tokens[idx_2], el))
        min_prob = np.min([el[2] for el in weighted_edges])
        max_prob = np.max([el[2] for el in weighted_edges])
        weighted_edges = [(el[0], el[1], (el[2] - mean_prob) / (max_prob - mean_prob)) for el in weighted_edges]

        G = nx.Graph()
        G.add_nodes_from([el for el in tokens])
        G.add_weighted_edges_from(weighted_edges)

        plot = Plot(plot_width=500, plot_height=500,
                    x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))
        plot.add_tools(HoverTool(tooltips=None), TapTool(), BoxSelectTool())

        graph_renderer = from_networkx(G, nx.circular_layout, scale=1, center=(0, 0))

        graph_renderer.node_renderer.data_source.data['colors'] = Spectral8[:len(tokens)]
        graph_renderer.node_renderer.glyph = Circle(size=15, line_color=None, fill_color="colors")
        graph_renderer.node_renderer.selection_glyph = Circle(size=15, fill_color="colors")
        graph_renderer.node_renderer.hover_glyph = Circle(size=15, fill_color="grey")

        graph_renderer.edge_renderer.data_source.data["line_width"] = [G.get_edge_data(a, b)['weight'] * 3 for a, b in
                                                                       G.edges()]
        graph_renderer.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_width={'field': 'line_width'})
        graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color="grey", line_width=5)
        graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color="grey", line_width=5)

        graph_renderer.selection_policy = NodesAndLinkedEdges()
        graph_renderer.inspection_policy = EdgesAndLinkedNodes()

        plot.renderers.append(graph_renderer)

        x, y = zip(*graph_renderer.layout_provider.graph_layout.values())
        data = {'x': list(x), 'y': list(y), 'connectionNames': tokens}
        source = ColumnDataSource(data)
        labels = LabelSet(x='x', y='y', text='connectionNames', source=source, text_align='center')
        plot.renderers.append(labels)
        plot.add_tools(SaveTool())
        show(plot)



model = BERTEmbeddingEval()
model.visualize_sentences("배", ["배가 고파서 밥 먹었어", "사과와 배는 맛있어"])
model.visualize_self_attention_scores("배고파 밥줘")