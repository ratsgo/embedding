import sys
sys.path.append('models')

import tensorflow as tf
from bert.modeling import BertModel, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from bilm import Batcher, BidirectionalLanguageModel, weight_layers
from preprocess import get_tokenizer, post_processing

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

from bokeh.io import show
from bokeh.plotting import figure
from bokeh.models import Plot, Range1d, MultiLine, Circle, HoverTool, TapTool, BoxSelectTool, LinearColorMapper, ColumnDataSource, LabelSet, SaveTool, ColorBar, BasicTicker
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.palettes import Spectral8


class SentenceEmbeddingEvaluator:

    def __init__(self, model_name, dimension):
        # reset graphs.
        tf.reset_default_graph()
        self.model_name = model_name
        self.dimension = dimension

    def get_token_vector_sequence(self, sentence):
        raise NotImplementedError

    def get_sentence_vector(self, sentence):
        raise NotImplementedError

    def predict(self, sentence):
        raise NotImplementedError

    def tokenize(self, sentence):
        raise NotImplementedError

    def make_input(self, tokens):
        raise NotImplementedError

    def visualize_homonym(self, homonym, sentences, palette="Viridis256"):
        """
            Visualize homonyms (2d vector space)
            Inspired by:
            https://github.com/hengluchang/visualizing_contextual_vectors/blob/master/elmo_vis.py
        """
        tokenized_sentences = []
        vecs = np.zeros((1, self.dimension))
        for sentence in sentences:
            tokens, vec = self.get_token_vector_sequence(sentence)
            tokenized_sentences.append(tokens)
            vecs = np.concatenate([vecs, vec], axis=0)
        # process sentences
        token_list, processed_sentences = [], []
        for tokens in tokenized_sentences:
            token_list.extend(tokens)
            sentence = []
            for token in tokens:
                if self.model_name == "bert":
                    processed_token = token.replace("##", "")
                else:
                    processed_token = token
                if token == homonym:
                    processed_token = "\"" + processed_token + "\""
                sentence.append(processed_token)
            processed_sentences.append(' '.join(sentence))
        # dimension reduction
        tsne = TSNE(n_components=2)
        tsne_results = tsne.fit_transform(vecs[1:])
        # only plot the word representation of interest
        interest_vecs, idx = np.zeros((len(sentences), 2)), 0
        for word, vec in zip(token_list, tsne_results):
            if word == homonym:
                interest_vecs[idx] = vec
                idx += 1
        df = pd.DataFrame(columns=['x', 'y', 'annotation'])
        df['x'], df['y'], df['annotation'] = interest_vecs[:, 0], interest_vecs[:, 1], processed_sentences
        source = ColumnDataSource(ColumnDataSource.from_df(df))
        labels = LabelSet(x="x", y="y", text="annotation", y_offset=8,
                          text_font_size="12pt", text_color="#555555",
                          source=source, text_align='center')
        color_mapper = LinearColorMapper(palette=palette, low=min(tsne_results[:, 1]), high=max(tsne_results[:, 1]))
        plot = figure(plot_width=900, plot_height=900)
        plot.scatter("x", "y", size=12, source=source, color={'field': 'y', 'transform': color_mapper},
                     line_color=None,
                     fill_alpha=0.8)
        plot.add_layout(labels)
        show(plot, notebook_handle=True)

    def visualize_sentences(self, sentences, palette="Viridis256"):
        vecs = np.array([self.get_sentence_vector(sentence)[1] for sentence in sentences])
        tsne = TSNE(n_components=2)
        tsne_results = tsne.fit_transform(vecs)
        df = pd.DataFrame(columns=['x', 'y', 'sentence'])
        df['x'], df['y'], df['sentence'] = tsne_results[:, 0], tsne_results[:, 1], sentences
        source = ColumnDataSource(ColumnDataSource.from_df(df))
        labels = LabelSet(x="x", y="y", text="sentence", y_offset=8,
                          text_font_size="8pt", text_color="#555555",
                          source=source, text_align='center')
        color_mapper = LinearColorMapper(palette=palette, low=min(tsne_results[:, 1]), high=max(tsne_results[:, 1]))
        plot = figure(plot_width=900, plot_height=900)
        plot.scatter("x", "y", size=12, source=source, color={'field': 'y', 'transform': color_mapper}, line_color=None, fill_alpha=0.8)
        plot.add_layout(labels)
        show(plot, notebook_handle=True)

    def visualize_between_sentences(self, sentences, palette="Viridis256"):
        vec_list, df_list, score_list = [], [], []
        for sentence in sentences:
            _, vec = self.get_sentence_vector(sentence)
            vec_list.append(vec)
        for sent1_idx, sentence1 in enumerate(sentences):
            for sent2_idx, sentence2 in enumerate(sentences):
                vec1, vec2 = vec_list[sent1_idx], vec_list[sent2_idx]
                if np.any(vec1) and np.any(vec2):
                    score = cosine_similarity(X=[vec1], Y=[vec2])
                    df_list.append({'x': sentence1, 'y': sentence2, 'similarity': score[0][0]})
                    score_list.append(score[0][0])
        df = pd.DataFrame(df_list)
        color_mapper = LinearColorMapper(palette=palette, low=np.max(score_list), high=np.min(score_list))
        TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"
        p = figure(x_range=sentences, y_range=list(reversed(sentences)),
                   x_axis_location="above", plot_width=900, plot_height=900,
                   toolbar_location='below', tools=TOOLS,
                   tooltips=[('sentences', '@x @y'), ('similarity', '@similarity')])
        p.grid.grid_line_color = None
        p.axis.axis_line_color = None
        p.axis.major_tick_line_color = None
        p.axis.major_label_standoff = 0
        p.xaxis.major_label_orientation = 3.14 / 3
        p.rect(x="x", y="y", width=1, height=1,
               source=df,
               fill_color={'field': 'similarity', 'transform': color_mapper},
               line_color=None)
        color_bar = ColorBar(ticker=BasicTicker(desired_num_ticks=5),
                             color_mapper=color_mapper, major_label_text_font_size="7pt",
                             label_standoff=6, border_line_color=None, location=(0, 0))
        p.add_layout(color_bar, 'right')
        show(p)


class BERTEmbeddingEval(SentenceEmbeddingEvaluator):

    def __init__(self, model_fname="data/bert",
                 bertconfig_fname="data/bert/multi_cased_L-12_H-768_A-12/bert_config.json",
                 vocab_fname="data/bert/multi_cased_L-12_H-768_A-12/vocab.txt",
                 max_seq_length=32, dimension=768):

        super().__init__("bert", dimension)
        config = BertConfig.from_json_file(bertconfig_fname)
        self.tokenizer = BertTokenizer(vocab_file=vocab_fname, do_lower_case=False)
        self.max_seq_length = max_seq_length
        self.input_ids = tf.placeholder(tf.int32, [1, max_seq_length], name='inputs_ids')
        self.input_mask = tf.placeholder(tf.int32, [1, max_seq_length], name='input_mask')
        self.segment_ids = tf.placeholder(tf.int32, [1, max_seq_length], name='segment_ids')
        self.model = BertModel(config=config,
                               is_training=False,
                               input_ids=self.input_ids,
                               input_mask=self.input_mask,
                               token_type_ids=self.segment_ids)
        logits = tf.contrib.layers.fully_connected(inputs=self.model.pooled_output,
                                                   num_outputs=2,
                                                   activation_fn=None,
                                                   weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                   biases_initializer=tf.zeros_initializer())
        self.probs = tf.nn.softmax(logits, axis=-1, name='probs')
        saver = tf.train.Saver(tf.global_variables())
        self.sess = tf.Session()
        checkpoint_path = tf.train.latest_checkpoint(model_fname)
        saver.restore(self.sess, checkpoint_path)

    def predict(self, sentence):
        tokens = self.tokenize(sentence)
        model_input = self.make_input(tokens)
        probs = self.sess.run(self.probs, model_input)
        return probs

    """
    sentence를 입력하면 토크나이즈 결과와 token 벡터 시퀀스를 반환한다
        - shape :[[# of tokens], [batch size, max seq length, dimension]]
    """
    def get_token_vector_sequence(self, sentence):
        tokens = self.tokenize(sentence)
        model_input = self.make_input(tokens)
        return [tokens, self.sess.run(self.model.get_sequence_output()[0], model_input)[:len(tokens)]]

    """
    sentence를 입력하면 토크나이즈 결과와 [CLS] 벡터를 반환한다
         - shape :[[# of tokens], [batch size, dimension]]
    """
    def get_sentence_vector(self, sentence):
        tokens = self.tokenize(sentence)
        model_input = self.make_input(tokens)
        return [tokens, self.sess.run(self.model.pooled_output, model_input)[0]]

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



class ELMoEmbeddingEval(SentenceEmbeddingEvaluator):

    def __init__(self, tune_model_fname="data/elmo",
                 pretrain_model_fname="data/elmo/elmo.model",
                 options_fname="data/elmo/options.json",
                 vocab_fname="data/elmo/elmo-vocab.txt",
                 max_characters_per_token=30, dimension=256):

        # configurations
        super().__init__("elmo", dimension)
        self.tokenizer = get_tokenizer("mecab")
        self.batcher = Batcher(lm_vocab_file=vocab_fname, max_token_length=max_characters_per_token)
        # Load pretrained ELMo model.
        bilm = BidirectionalLanguageModel(options_fname, pretrain_model_fname)
        # Input placeholders to the biLM.
        self.ids_placeholder = tf.placeholder(tf.int32, shape=(None, None, max_characters_per_token), name='input')
        # Get ops to compute the LM embeddings.
        embeddings_op = bilm(self.ids_placeholder)
        # the ELMo layer
        # shape : [batch_size, unroll_steps, dimension]
        self.elmo_embeddings = weight_layers("elmo_embeddings",
                                             embeddings_op,
                                             l2_coef=0.0,
                                             use_top_only=False,
                                             do_layer_norm=True)["weighted_op"]
        # Bidirectional LSTM with Attention Layer
        lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=512,
                                               cell_clip=5,
                                               proj_clip=5)
        lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=512,
                                               cell_clip=5,
                                               proj_clip=5)
        lstm_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_fw,
                                                         cell_bw=lstm_cell_bw,
                                                         inputs=self.elmo_embeddings,
                                                         sequence_length=embeddings_op['lengths'],
                                                         dtype=tf.float32)
        # Attention Layer
        output_fw, output_bw = lstm_output
        # (batch_size, seq_len, HIDDEN_SIZE)
        H = tf.nn.tanh(output_fw + output_bw)
        # softmax(dot(W, H)) : (batch_size, seq_len, 1)
        attention_score = tf.nn.softmax(tf.contrib.layers.fully_connected(inputs=H, num_outputs=1, activation_fn=None))
        # dot(prob, H) : (batch_size, HIDDEN_SIZE, 1) > (batch_size, HIDDEN_SIZE)
        attention_output = tf.squeeze(tf.matmul(tf.transpose(H, perm=[0, 2, 1]), attention_score), axis=-1)
        layer_output = tf.nn.tanh(attention_output)
        # Feed-Forward Layer
        fc = tf.contrib.layers.fully_connected(inputs=layer_output,
                                               num_outputs=512,
                                               activation_fn=tf.nn.relu,
                                               weights_initializer=tf.contrib.layers.xavier_initializer(),
                                               biases_initializer=tf.zeros_initializer())
        logits = tf.contrib.layers.fully_connected(inputs=fc,
                                                   num_outputs=2,
                                                   activation_fn=None,
                                                   weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                   biases_initializer=tf.zeros_initializer())
        self.probs = tf.nn.softmax(logits, axis=-1, name='probs')
        # restore model
        saver = tf.train.Saver(tf.global_variables())
        self.sess = tf.Session()
        checkpoint_path = tf.train.latest_checkpoint(tune_model_fname)
        saver.restore(self.sess, checkpoint_path)

    def predict(self, sentence):
        tokens = self.tokenize(sentence)
        model_input = self.make_input(tokens)
        probs = self.sess.run(self.probs, model_input)
        return probs

    """
    sentence를 입력하면 토크나이즈 결과와 token 벡터 시퀀스를 반환한다
        - shape :[[# of tokens], [batch size, max seq length, dimension]]
    """
    def get_token_vector_sequence(self, sentence):
        tokens = self.tokenize(sentence)
        model_input = self.make_input(tokens)
        sentence_vector = self.sess.run(self.elmo_embeddings, model_input)
        return [tokens, sentence_vector[0]]

    """
    sentence를 입력하면 토크나이즈 결과와 토큰 시퀀스의 마지막 벡터를 반환한다
    ELMo는 Language Model이기 때문에 토큰 시퀀스 마지막 벡터에 많은 정보가 녹아 있다
         - shape :[[# of tokens], [batch size, dimension]]
    """
    def get_sentence_vector(self, sentence):
        tokens, vecs = self.get_token_vector_sequence(sentence)
        return [tokens, vecs[-1]]

    def tokenize(self, sentence):
        tokens = self.tokenizer.morphs(sentence)
        return post_processing(tokens)

    def make_input(self, tokens):
        model_input = self.batcher.batch_sentences([tokens])
        input_feed = {self.ids_placeholder: model_input}
        return input_feed


import csv, random
sentences = []
with open("data/kor_pair_train.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter=",")
    next(reader) # skip head line
    for line in reader:
        _, _, _, sent1, sent2, _ = line
        sentences.append(sent1)
        sentences.append(sent2)
sampled_sentences = random.sample(sentences, 30)

model = BERTEmbeddingEval()
model.get_sentence_vector("나는 학교에 간다")
model.get_token_vector_sequence("나는 학교에 간다")
model.visualize_homonym("배", ["배가 고파서 밥 먹었어", "배가 아파서 병원에 갔어",  "고기를 많이 먹으면 배가 나온다",
                                "사과와 배는 맛있어", "갈아만든 배", "감기에 걸렸을 땐 배를 달여 드세요",
                                "항구에 배가 많다", "배를 타면 멀미가 난다", "배를 건조하는 데 돈이 많이 든다"])
model.visualize_self_attention_scores("배가 아파서 병원에 갔어")
model.predict("이 영화 정말 재미 있다")
model.visualize_between_sentences(sampled_sentences)
model.visualize_sentences(sampled_sentences)

model = ELMoEmbeddingEval()
model.get_sentence_vector("나는 학교에 간다")
model.get_token_vector_sequence("나는 학교에 간다")
model.visualize_homonym("배", ["배가 고파서 밥 먹었어", "배가 아파서 병원에 갔어",  "고기를 많이 먹으면 배가 나온다",
                                "사과와 배는 맛있어", "갈아만든 배", "감기에 걸렸을 땐 배를 달여 드세요",
                                "항구에 배가 많다", "배를 타면 멀미가 난다", "배를 건조하는 데 돈이 많이 든다"])
model.predict("이 영화 정말 재미 있다")
model.visualize_between_sentences(sampled_sentences)
model.visualize_sentences(sampled_sentences)
