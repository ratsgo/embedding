import numpy as np
import pandas as pd
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

from bokeh.io import export_png, output_notebook, show
from bokeh.plotting import figure
from bokeh.models import Plot, Range1d, MultiLine, Circle, HoverTool, TapTool, BoxSelectTool, LinearColorMapper, ColumnDataSource, LabelSet, SaveTool, ColorBar, BasicTicker
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.palettes import Spectral8


def visualize_sentences(vecs, sentences, palette="Viridis256", filename="/notebooks/embedding/sentences.png",
                        use_notebook=False):
    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(vecs)
    df = pd.DataFrame(columns=['x', 'y', 'sentence'])
    df['x'], df['y'], df['sentence'] = tsne_results[:, 0], tsne_results[:, 1], sentences
    source = ColumnDataSource(ColumnDataSource.from_df(df))
    labels = LabelSet(x="x", y="y", text="sentence", y_offset=8,
                      text_font_size="12pt", text_color="#555555",
                      source=source, text_align='center')
    color_mapper = LinearColorMapper(palette=palette, low=min(tsne_results[:, 1]), high=max(tsne_results[:, 1]))
    plot = figure(plot_width=900, plot_height=900)
    plot.scatter("x", "y", size=12, source=source, color={'field': 'y', 'transform': color_mapper}, line_color=None, fill_alpha=0.8)
    plot.add_layout(labels)
    if use_notebook:
        output_notebook()
        show(plot)
    else:
        export_png(plot, filename)
        print("save @ " + filename)


"""
Visualize homonyms (2d vector space)
Inspired by:
    https://github.com/hengluchang/visualizing_contextual_vectors/blob/master/elmo_vis.py
"""
def visualize_homonym(homonym, tokenized_sentences, vecs, model_name, palette="Viridis256",
                      filename="/notebooks/embedding/homonym.png", use_notebook=False):
    # process sentences
    token_list, processed_sentences = [], []
    for tokens in tokenized_sentences:
        token_list.extend(tokens)
        sentence = []
        for token in tokens:
            if model_name == "bert":
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
    interest_vecs, idx = np.zeros((len(tokenized_sentences), 2)), 0
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
    if use_notebook:
        output_notebook()
        show(plot)
    else:
        export_png(plot, filename)
        print("save @ " + filename)


def visualize_between_sentences(sentences, vec_list, palette="Viridis256",
                                filename="/notebooks/embedding/between-sentences.png",
                                use_notebook=False):
    df_list, score_list = [], []
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
    if use_notebook:
        output_notebook()
        show(p)
    else:
        export_png(p, filename)
        print("save @ " + filename)


def visualize_self_attention_scores(tokens, scores, filename="/notebooks/embedding/self-attention.png",
                                    use_notebook=False):
    mean_prob = np.mean(scores)
    weighted_edges = []
    for idx_1, token_prob_dist_1 in enumerate(scores):
        for idx_2, el in enumerate(token_prob_dist_1):
            if idx_1 == idx_2 or el < mean_prob:
                weighted_edges.append((tokens[idx_1], tokens[idx_2], 0))
            else:
                weighted_edges.append((tokens[idx_1], tokens[idx_2], el))
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
    if use_notebook:
        output_notebook()
        show(plot)
    else:
        export_png(plot, filename)
        print("save @ " + filename)


def visualize_words(words, vecs, palette="Viridis256", filename="/notebooks/embedding/words.png",
                    use_notebook=False):
    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(vecs)
    df = pd.DataFrame(columns=['x', 'y', 'word'])
    df['x'], df['y'], df['word'] = tsne_results[:, 0], tsne_results[:, 1], list(words)
    source = ColumnDataSource(ColumnDataSource.from_df(df))
    labels = LabelSet(x="x", y="y", text="word", y_offset=8,
                      text_font_size="15pt", text_color="#555555",
                      source=source, text_align='center')
    color_mapper = LinearColorMapper(palette=palette, low=min(tsne_results[:, 1]), high=max(tsne_results[:, 1]))
    plot = figure(plot_width=900, plot_height=900)
    plot.scatter("x", "y", size=12, source=source, color={'field': 'y', 'transform': color_mapper}, line_color=None,
                 fill_alpha=0.8)
    plot.add_layout(labels)
    if use_notebook:
        output_notebook()
        show(plot)
    else:
        export_png(plot, filename)
        print("save @ " + filename)


def visualize_between_words(words, vecs, palette="Viridis256", filename="/notebooks/embedding/between-words.png",
                            use_notebook=False):
    df_list = []
    for word1_idx, word1 in enumerate(words):
        for word2_idx, word2 in enumerate(words):
            vec1 = vecs[word1_idx]
            vec2 = vecs[word2_idx]
            if np.any(vec1) and np.any(vec2):
                score = cosine_similarity(X=[vec1], Y=[vec2])
                df_list.append({'x': word1, 'y': word2, 'similarity': score[0][0]})
    df = pd.DataFrame(df_list)
    color_mapper = LinearColorMapper(palette=palette, low=1, high=0)
    TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"
    p = figure(x_range=list(words), y_range=list(reversed(list(words))),
               x_axis_location="above", plot_width=900, plot_height=900,
               toolbar_location='below', tools=TOOLS,
               tooltips=[('words', '@x @y'), ('similarity', '@similarity')])
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
    if use_notebook:
        output_notebook()
        show(p)
    else:
        export_png(p, filename)
        print("save @ " + filename)
