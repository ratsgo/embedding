import sys
sys.path.append('models')
from bert.modeling import BertModel, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
import tensorflow as tf
import numpy as np

def make_input(tokens):
    tokens = tokens[:(max_seq_length - 2)]
    token_sequence = ["[CLS]"] + tokens + ["[SEP]"]
    segment = [0] * len(token_sequence)
    sequence = tokenizer.convert_tokens_to_ids(token_sequence)
    current_length = len(sequence)
    padding_length = max_seq_length - current_length
    input_feed = {
        input_ids: np.array([sequence + [0] * padding_length]),
        segment_ids: np.array([segment + [0] * padding_length]),
        input_mask: np.array([[1] * current_length + [0] * padding_length])
    }
    return input_feed

max_seq_length = 32
bertconfig_fname = "data/bert/multi_cased_L-12_H-768_A-12/bert_config.json"
vocab_fname = "data/bert/multi_cased_L-12_H-768_A-12/vocab.txt"
model_fname = "data/bert"

config = BertConfig.from_json_file(bertconfig_fname)
tokenizer = BertTokenizer(vocab_file=vocab_fname, do_lower_case=False)
sentence = "배고파 밥줘"
tokens = tokenizer.tokenize(sentence)

input_ids = tf.placeholder(tf.int32, [1, max_seq_length], name='inputs_ids')
input_mask = tf.placeholder(tf.int32, [1, max_seq_length], name='input_mask')
segment_ids = tf.placeholder(tf.int32, [1, max_seq_length], name='segment_ids')

model = BertModel(config=config,
                  is_training=False,
                  input_ids=input_ids,
                  input_mask=input_mask,
                  token_type_ids=segment_ids)

saver = tf.train.Saver(tf.global_variables())
sess = tf.Session()
checkpoint_path = tf.train.latest_checkpoint(model_fname)
saver.restore(sess, checkpoint_path)

a = sess.run(model.attn_probs_for_visualization_list, make_input(tokens))
# a의 length : 12개 (레이어별)
# 마지막 레이어의 attention scores : a[-1], shape=[batch_size, num_attention_heads, max_seq_length, max_seq_length]
# 마지막 레이어를 취한 뒤, attention head 기준으로 sum
b = np.sum(a[-1][0], axis=0)
# b에서 토큰 개수만큼 취함
attn_probs = b[:len(tokens), :len(tokens)]

import networkx as nx

from bokeh.io import show
from bokeh.models import Plot, Range1d, MultiLine, Circle, HoverTool, TapTool, BoxSelectTool, LinearColorMapper, ColumnDataSource, LabelSet, SaveTool
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.palettes import Spectral8

def visualize(tokens, attn_probs, palette="Viridis256"):
    mean_prob = np.mean(attn_probs)
    weighted_edges = []
    for idx_1, token_prob_dist_1 in enumerate(attn_probs):
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

    graph_renderer.edge_renderer.data_source.data["line_width"] = [G.get_edge_data(a, b)['weight'] * 3 for a, b in G.edges()]
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