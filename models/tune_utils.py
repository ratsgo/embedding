import sys, os, random, argparse, re, collections
import numpy as np
import tensorflow as tf
from tensorflow.contrib import nccl
from gensim.models import Word2Vec
from collections import defaultdict
from scipy.stats import truncnorm
import sentencepiece as spm

sys.path.append('models')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocess import get_tokenizer, post_processing
from bilm import Batcher, BidirectionalLanguageModel, weight_layers
from bert.modeling import BertModel, BertConfig
from bert.optimization import create_optimizer
from bert.tokenization import FullTokenizer, convert_to_unicode
from xlnet.modeling import classification_loss
from xlnet.xlnet import XLNetConfig, RunConfig, XLNetModel
from xlnet.prepro_utils import preprocess_text, encode_pieces, encode_ids
from xlnet.model_utils import AdamWeightDecayOptimizer


def make_xlnet_graph(input_ids, input_mask, segment_ids, label_ids, model_config_path,
                     num_labels, is_training_placeholder, tune=False):
    xlnet_config = XLNetConfig(json_path=model_config_path)
    # 모두 기본값으로 세팅
    kwargs = dict(
        is_training=is_training_placeholder,
        use_tpu=False,
        use_bfloat16=False,
        dropout=0.1,
        dropatt=0.1,
        init="normal",
        init_range=0.1,
        init_std=0.1,
        clamp_len=-1)
    run_config = RunConfig(**kwargs)
    xlnet_model = XLNetModel(
        xlnet_config=xlnet_config,
        run_config=run_config,
        input_ids=input_ids,
        seg_ids=segment_ids,
        input_mask=input_mask)
    # summary_type="last", 마지막 레이어 히든 벡터 시퀀스의 마지막 벡터
    # summary_type="first", 마지막 레이어 히든 벡터 시퀀스의 첫번째 벡터
    # summary_type="mean", 마지막 레이어 히든 벡터 시퀀스의 평균 벡터
    # summary_type="attn", 마지막 레이어 히든 벡터 시퀀스에 멀티 헤드 어텐션 적용
    # use_proj=True, 이미 만든 summary 벡터에 선형변환 + tanh 적용 (BERT와 동일)
    # use_proj=False, 이미 만든 summary 벡터를 그대로 리턴
    summary = xlnet_model.get_pooled_out(summary_type="last", use_summ_proj=True)
    # summary 벡터에 활성함수(act_fn) 없이 선형변환 후 cross entropy loss 구함
    per_example_loss, logits = classification_loss(
        hidden=summary,
        labels=label_ids,
        n_class=num_labels,
        initializer=xlnet_model.get_initializer(),
        scope="classification_layer",
        return_logits=True)
    if tune:
        # loss layer
        total_loss = tf.reduce_mean(per_example_loss)
        return logits, total_loss
    else:
        # prob Layer
        probs = tf.nn.softmax(logits, axis=-1, name='probs')
        return probs


def make_elmo_graph(options_fname, pretrain_model_fname, max_characters_per_token, num_labels, tune=False):
    """
        ids_placeholder : ELMo 네트워크의 입력값 (ids)
            - shape : [batch_size, unroll_steps, max_character_byte_length]
        elmo_embeddings : fine tuning 네트워크의 입력값 (ELMo 네트워크의 출력값)
            - shape : [batch_size, unroll_steps, dimension]
        labels_placeholder : fine tuning 네트워크의 출력값 (예 : 긍정=1/부정=0)
            - shape : [batch_size]
        loss : fine tuning 네트워크의 loss
    """
    # Build the biLM graph.
    # Load pretrained ELMo model.
    bilm = BidirectionalLanguageModel(options_fname, pretrain_model_fname)
    # Input placeholders to the biLM.
    ids_placeholder = tf.placeholder(tf.int32, shape=(None, None, max_characters_per_token), name='input')
    if tune:
        # Output placeholders to the fine-tuned Net.
        labels_placeholder = tf.placeholder(tf.int32, shape=(None))
    else:
        labels_placeholder = None
    # Get ops to compute the LM embeddings.
    embeddings_op = bilm(ids_placeholder)
    # Get lengths.
    input_lengths = embeddings_op['lengths']
    # define dropout
    if tune:
        dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
    else:
        dropout_keep_prob = tf.constant(1.0, dtype=tf.float32)
    # the ELMo layer
    # shape : [batch_size, unroll_steps, dimension]
    elmo_embeddings = weight_layers("elmo_embeddings",
                                    embeddings_op,
                                    l2_coef=0.0,
                                    use_top_only=False,
                                    do_layer_norm=True)
    # input of fine tuning network
    features = tf.nn.dropout(elmo_embeddings['weighted_op'], dropout_keep_prob)
    # Bidirectional LSTM Layer
    lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=512,
                                           cell_clip=5,
                                           proj_clip=5)
    lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=512,
                                           cell_clip=5,
                                           proj_clip=5)
    lstm_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_fw,
                                                     cell_bw=lstm_cell_bw,
                                                     inputs=features,
                                                     sequence_length=input_lengths,
                                                     dtype=tf.float32)

    # Attention Layer
    output_fw, output_bw = lstm_output
    H = tf.contrib.layers.fully_connected(inputs=output_fw + output_bw, num_outputs=256, activation_fn=tf.nn.tanh)
    attention_score = tf.nn.softmax(tf.contrib.layers.fully_connected(inputs=H, num_outputs=1, activation_fn=None), axis=1)
    attention_output = tf.squeeze(tf.matmul(tf.transpose(H, perm=[0, 2, 1]), attention_score), axis=-1)
    layer_output = tf.nn.dropout(attention_output, dropout_keep_prob)

    # Feed-Forward Layer
    fc = tf.contrib.layers.fully_connected(inputs=layer_output,
                                           num_outputs=512,
                                           activation_fn=tf.nn.relu,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           biases_initializer=tf.zeros_initializer())
    features_drop = tf.nn.dropout(fc, dropout_keep_prob)
    logits = tf.contrib.layers.fully_connected(inputs=features_drop,
                                               num_outputs=num_labels,
                                               activation_fn=None,
                                               weights_initializer=tf.contrib.layers.xavier_initializer(),
                                               biases_initializer=tf.zeros_initializer())
    if tune:
        # Loss Layer
        CE = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=logits)
        loss = tf.reduce_mean(CE)
        return ids_placeholder, labels_placeholder, dropout_keep_prob, logits, loss
    else:
        # prob Layer
        probs = tf.nn.softmax(logits, axis=-1, name='probs')
        return ids_placeholder, elmo_embeddings, probs


def make_bert_graph(bert_config, max_seq_length, dropout_keep_prob_rate, num_labels, tune=False):
    input_ids = tf.placeholder(tf.int32, [None, max_seq_length], name='inputs_ids')
    input_mask = tf.placeholder(tf.int32, [None, max_seq_length], name='input_mask')
    segment_ids = tf.placeholder(tf.int32, [None, max_seq_length], name='segment_ids')
    model = BertModel(config=bert_config,
                      is_training=tune,
                      input_ids=input_ids,
                      input_mask=input_mask,
                      token_type_ids=segment_ids)
    if tune:
        bert_embeddings_dropout = tf.nn.dropout(model.pooled_output, keep_prob=(1 - dropout_keep_prob_rate))
        label_ids = tf.placeholder(tf.int32, [None], name='label_ids')
    else:
        bert_embeddings_dropout = model.pooled_output
        label_ids = None
    logits = tf.contrib.layers.fully_connected(inputs=bert_embeddings_dropout,
                                               num_outputs=num_labels,
                                               activation_fn=None,
                                               weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                               biases_initializer=tf.zeros_initializer())
    if tune:
        # loss layer
        CE = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_ids, logits=logits)
        loss = tf.reduce_mean(CE)
        return input_ids, input_mask, segment_ids, label_ids, logits, loss
    else:
        # prob layer
        probs = tf.nn.softmax(logits, axis=-1, name='probs')
        return model, input_ids, input_mask, segment_ids, probs


def make_word_embedding_graph(num_labels, vocab_size, embedding_size, tune=False):
    ids_placeholder = tf.placeholder(tf.int32, [None, None], name="input_ids")
    input_lengths = tf.placeholder(tf.int32, [None], name="input_lengths")
    labels_placeholder = tf.placeholder(tf.int32, [None], name="label_ids")
    if tune:
        dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
    else:
        dropout_keep_prob = tf.constant(1.0, dtype=tf.float32)
    We = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_size]), trainable=True)
    embedding_placeholder = tf.placeholder(tf.float32, shape=[vocab_size, embedding_size])
    embed_init = We.assign(embedding_placeholder)
    # shape : [batch_size, unroll_steps, dimension]
    embedded_words = tf.nn.embedding_lookup(We, ids_placeholder)
    # input of fine tuning network
    features = tf.nn.dropout(embedded_words, dropout_keep_prob)
    # Bidirectional LSTM Layer
    lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=embedding_size,
                                           cell_clip=5,
                                           proj_clip=5)
    lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=embedding_size,
                                           cell_clip=5,
                                           proj_clip=5)
    lstm_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_fw,
                                                     cell_bw=lstm_cell_bw,
                                                     inputs=features,
                                                     sequence_length=input_lengths,
                                                     dtype=tf.float32)

    # Attention Layer
    output_fw, output_bw = lstm_output
    H = tf.contrib.layers.fully_connected(inputs=output_fw + output_bw, num_outputs=256, activation_fn=tf.nn.tanh)
    attention_score = tf.nn.softmax(tf.contrib.layers.fully_connected(inputs=H, num_outputs=1, activation_fn=None), axis=1)
    attention_output = tf.squeeze(tf.matmul(tf.transpose(H, perm=[0, 2, 1]), attention_score), axis=-1)
    layer_output = tf.nn.dropout(attention_output, dropout_keep_prob)

    # Feed-Forward Layer
    fc = tf.contrib.layers.fully_connected(inputs=layer_output,
                                           num_outputs=512,
                                           activation_fn=tf.nn.relu,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           biases_initializer=tf.zeros_initializer())
    features_drop = tf.nn.dropout(fc, dropout_keep_prob)
    logits = tf.contrib.layers.fully_connected(inputs=features_drop,
                                               num_outputs=num_labels,
                                               activation_fn=None,
                                               weights_initializer=tf.contrib.layers.xavier_initializer(),
                                               biases_initializer=tf.zeros_initializer())
    if tune:
        # Loss Layer
        CE = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=logits)
        loss = tf.reduce_mean(CE)
        return ids_placeholder, input_lengths, labels_placeholder, dropout_keep_prob, embedding_placeholder, embed_init, logits, loss
    else:
        # prob Layer
        probs = tf.nn.softmax(logits, axis=-1, name='probs')
        return ids_placeholder, input_lengths, labels_placeholder, probs


class Tuner(object):

    def __init__(self, train_corpus_fname=None, tokenized_train_corpus_fname=None,
                 test_corpus_fname=None, tokenized_test_corpus_fname=None,
                 model_name="bert", model_save_path=None, vocab_fname=None, eval_every=1000,
                 batch_size=32, num_epochs=10, dropout_keep_prob_rate=0.9, model_ckpt_path=None,
                 sp_model_path=None):
        # configurations
        tf.logging.set_verbosity(tf.logging.INFO)
        self.model_name = model_name
        self.eval_every = eval_every
        self.model_ckpt_path = model_ckpt_path
        self.model_save_path = model_save_path
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.dropout_keep_prob_rate = dropout_keep_prob_rate
        self.best_valid_score = 0.0
        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)
        # define tokenizer
        if self.model_name == "bert":
            self.tokenizer = FullTokenizer(vocab_file=vocab_fname, do_lower_case=False)
        elif self.model_name == "xlnet":
            sp = spm.SentencePieceProcessor()
            sp.Load(sp_model_path)
            self.tokenizer = sp
        else:
            self.tokenizer = get_tokenizer("mecab")
        # load or tokenize corpus
        self.train_data, self.train_data_size = self.load_or_tokenize_corpus(train_corpus_fname, tokenized_train_corpus_fname)
        self.test_data, self.test_data_size = self.load_or_tokenize_corpus(test_corpus_fname, tokenized_test_corpus_fname)

    def load_or_tokenize_corpus(self, corpus_fname, tokenized_corpus_fname):
        data_set = []
        if os.path.exists(tokenized_corpus_fname):
            tf.logging.info("load tokenized corpus : " + tokenized_corpus_fname)
            with open(tokenized_corpus_fname, 'r') as f1:
                for line in f1:
                    tokens, label = line.strip().split("\u241E")
                    if len(tokens) > 0:
                        data_set.append([tokens.split(" "), int(label)])
        else:
            tf.logging.info("tokenize corpus : " + corpus_fname + " > " + tokenized_corpus_fname)
            with open(corpus_fname, 'r') as f2:
                next(f2)  # skip head line
                for line in f2:
                    sentence, label = line.strip().split("\u241E")
                    if self.model_name == "bert":
                        tokens = self.tokenizer.tokenize(sentence)
                    elif self.model_name == "xlnet":
                        normalized_sentence = preprocess_text(sentence, lower=False)
                        tokens = encode_pieces(self.tokenizer, normalized_sentence, return_unicode=False, sample=False)
                    else:
                        tokens = self.tokenizer.morphs(sentence)
                        tokens = post_processing(tokens)
                    if int(label) > 0.5:
                        int_label = 1
                    else:
                        int_label = 0
                    data_set.append([tokens, int_label])
            with open(tokenized_corpus_fname, 'w') as f3:
                for tokens, label in data_set:
                    f3.writelines(' '.join(tokens) + "\u241E" + str(label) + "\n")
        return data_set, len(data_set)

    def train(self, sess, saver, global_step, output_feed):
        train_batches = self.get_batch(self.train_data, num_epochs=self.num_epochs, is_training=True)
        checkpoint_loss = 0.0
        for current_input_feed in train_batches:
            _, _, _, current_loss = sess.run(output_feed, current_input_feed)
            checkpoint_loss += current_loss
            if global_step.eval(sess) % self.eval_every == 0:
                tf.logging.info("global step %d train loss %.4f" %
                                (global_step.eval(sess), checkpoint_loss / self.eval_every))
                checkpoint_loss = 0.0
                self.validation(sess, saver, global_step)

    def validation(self, sess, saver, global_step):
        valid_loss, valid_pred, valid_num_data = 0, 0, 0
        output_feed = [self.logits, self.loss]
        test_batches = self.get_batch(self.test_data, num_epochs=1, is_training=False)
        for current_input_feed, current_labels in test_batches:
            current_logits, current_loss = sess.run(output_feed, current_input_feed)
            current_preds = np.argmax(current_logits, axis=-1)
            valid_loss += current_loss
            valid_num_data += len(current_labels)
            for pred, label in zip(current_preds, current_labels):
                if pred == label:
                    valid_pred += 1
        valid_score = valid_pred / valid_num_data
        tf.logging.info("valid loss %.4f valid score %.4f" %
                        (valid_loss, valid_score))
        if valid_score > self.best_valid_score:
            self.best_valid_score = valid_score
            path = self.model_save_path + "/" + str(valid_score)
            saver.save(sess, path, global_step=global_step)

    def get_batch(self, data, num_epochs, is_training=True):
        if is_training:
            data_size = self.train_data_size
        else:
            data_size = self.test_data_size
        num_batches_per_epoch = int((data_size - 1) / self.batch_size)
        if is_training:
            tf.logging.info("num_batches_per_epoch : " + str(num_batches_per_epoch))
        for epoch in range(num_epochs):
            idx = random.sample(range(data_size), data_size)
            data = np.array(data)[idx]
            for batch_num in range(num_batches_per_epoch):
                batch_sentences = []
                batch_labels = []
                start_index = batch_num * self.batch_size
                end_index = (batch_num + 1) * self.batch_size
                features = data[start_index:end_index]
                for feature in features:
                    sentence, label = feature
                    batch_sentences.append(sentence)
                    batch_labels.append(int(label))
                yield self.make_input(batch_sentences, batch_labels, is_training)

    def make_input(self, sentences, labels, is_training):
        raise NotImplementedError

    def tune(self):
        raise NotImplementedError


class ELMoTuner(Tuner):

    def __init__(self, train_corpus_fname, test_corpus_fname,
                 vocab_fname, options_fname, pretrain_model_fname,
                 model_save_path, max_characters_per_token=30,
                 batch_size=32, num_labels=2):
        # Load a corpus.
        super().__init__(train_corpus_fname=train_corpus_fname,
                         tokenized_train_corpus_fname=train_corpus_fname + ".elmo-tokenized",
                         test_corpus_fname=test_corpus_fname,
                         tokenized_test_corpus_fname=test_corpus_fname + ".elmo-tokenized",
                         model_name="elmo", vocab_fname=vocab_fname,
                         model_save_path=model_save_path, batch_size=batch_size)
        # configurations
        self.options_fname = options_fname
        self.pretrain_model_fname = pretrain_model_fname
        self.max_characters_per_token = max_characters_per_token
        self.num_labels = 2 # positive, negative
        self.num_train_steps = (int((len(self.train_data) - 1) / self.batch_size) + 1) * self.num_epochs
        self.eval_every = int(self.num_train_steps / self.num_epochs)  # epoch마다 평가
        # Create a Batcher to map text to character ids.
        # lm_vocab_file = ELMo는 token vocab이 없어도 on-the-fly로 입력 id들을 만들 수 있다
        # 하지만 자주 나오는 char sequence, 즉 vocab을 미리 id로 만들어 놓으면 좀 더 빠른 학습이 가능
        # max_token_length = the maximum number of characters in each token
        self.batcher = Batcher(lm_vocab_file=vocab_fname, max_token_length=self.max_characters_per_token)
        self.training = tf.placeholder(tf.bool)
        # build train graph
        self.ids_placeholder, self.labels_placeholder, self.dropout_keep_prob, self.logits, self.loss = make_elmo_graph(options_fname,
                                                                                                                        pretrain_model_fname,
                                                                                                                        max_characters_per_token,
                                                                                                                        num_labels, tune=True)

    def tune(self):
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        output_feed = [train_op, global_step, self.logits, self.loss]
        saver = tf.train.Saver(max_to_keep=1)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        self.train(sess, saver, global_step, output_feed)

    def make_input(self, sentences, labels, is_training):
        current_input = self.batcher.batch_sentences(sentences)
        current_output = np.array(labels)
        if is_training:
            input_feed = {
                self.ids_placeholder: current_input,
                self.labels_placeholder: current_output,
                self.dropout_keep_prob: self.dropout_keep_prob_rate,
                self.training: True
            }
        else:
            input_feed_ = {
                self.ids_placeholder: current_input,
                self.labels_placeholder: current_output,
                self.dropout_keep_prob: 1.0,
                self.training: False
            }
            input_feed = [input_feed_, current_output]
        return input_feed


class BERTTuner(Tuner):

    def __init__(self, train_corpus_fname, test_corpus_fname, vocab_fname,
                 pretrain_model_fname, bertconfig_fname, model_save_path,
                 max_seq_length=128, warmup_proportion=0.1,
                 batch_size=32, learning_rate=2e-5, num_labels=2):
        # Load a corpus.
        super().__init__(train_corpus_fname=train_corpus_fname,
                         tokenized_train_corpus_fname=train_corpus_fname + ".bert-tokenized",
                         test_corpus_fname=test_corpus_fname, batch_size=batch_size,
                         tokenized_test_corpus_fname=test_corpus_fname + ".bert-tokenized",
                         model_name="bert", vocab_fname=vocab_fname, model_save_path=model_save_path)
        # configurations
        config = BertConfig.from_json_file(bertconfig_fname)
        self.pretrain_model_fname = pretrain_model_fname
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_labels = 2 # positive, negative
        self.PAD_INDEX = 0
        self.CLS_TOKEN = "[CLS]"
        self.SEP_TOKEN = "[SEP]"
        self.num_train_steps = (int((len(self.train_data) - 1) / self.batch_size) + 1) * self.num_epochs
        self.num_warmup_steps = int(self.num_train_steps * warmup_proportion)
        self.eval_every = int(self.num_train_steps / self.num_epochs)  # epoch마다 평가
        self.training = tf.placeholder(tf.bool)
        # build train graph
        self.input_ids, self.input_mask, self.segment_ids, self.label_ids, self.logits, self.loss = make_bert_graph(config,
                                                                                                                    max_seq_length,
                                                                                                                    self.dropout_keep_prob_rate,
                                                                                                                    num_labels, tune=True)

    def tune(self):
        global_step = tf.train.get_or_create_global_step()
        tf.logging.info("num_train_steps: " + str(self.num_train_steps))
        tf.logging.info("num_warmup_steps: " + str(self.num_warmup_steps))
        train_op = create_optimizer(self.loss, self.learning_rate, self.num_train_steps, self.num_warmup_steps, use_tpu=False)
        output_feed = [train_op, global_step, self.logits, self.loss]
        restore_vars = [v for v in tf.trainable_variables() if "bert" in v.name]
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        tf.train.Saver(restore_vars).restore(sess, self.pretrain_model_fname)
        saver = tf.train.Saver(max_to_keep=1)
        self.train(sess, saver, global_step, output_feed)

    def make_input(self, sentences, labels, is_training):
        collated_batch = {'sequences': [], 'segments': [], 'masks': []}
        for tokens in sentences:
            tokens = tokens[:(self.max_seq_length - 2)]
            token_sequence = [self.CLS_TOKEN] + tokens + [self.SEP_TOKEN]
            segment = [0] * len(token_sequence)
            sequence = self.tokenizer.convert_tokens_to_ids(token_sequence)
            current_length = len(sequence)
            padding_length = self.max_seq_length - current_length
            collated_batch['sequences'].append(sequence + [self.PAD_INDEX] * padding_length)
            collated_batch['segments'].append(segment + [self.PAD_INDEX] * padding_length)
            collated_batch['masks'].append([1] * current_length + [self.PAD_INDEX] * padding_length)
        if is_training:
            input_feed = {
                self.training: is_training,
                self.input_ids: np.array(collated_batch['sequences']),
                self.segment_ids: np.array(collated_batch['segments']),
                self.input_mask: np.array(collated_batch['masks']),
                self.label_ids: np.array(labels)
            }
        else:
            input_feed_ = {
                self.training: is_training,
                self.input_ids: np.array(collated_batch['sequences']),
                self.segment_ids: np.array(collated_batch['segments']),
                self.input_mask: np.array(collated_batch['masks']),
                self.label_ids: np.array(labels)
            }
            input_feed = [input_feed_, labels]
        return input_feed


class XLNetTuner(Tuner):

    def __init__(self, train_corpus_fname, test_corpus_fname,
                 pretrain_model_fname, config_fname, model_save_path,
                 sp_model_path, max_seq_length=64, warmup_steps=1000, decay_method="poly",
                 min_lr_ratio=0.0, adam_epsilon=1e-8, num_gpus=1, weight_decay=0.00,
                 batch_size=128, learning_rate=3e-5, clip=1.0, num_labels=2):
        # configurations
        self.pretrain_model_fname = pretrain_model_fname
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size * num_gpus
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.decay_method = decay_method
        self.min_lr_ratio = min_lr_ratio
        self.weight_decay = weight_decay
        self.adam_epsilon = adam_epsilon
        self.clip = clip
        self.num_labels = num_labels
        self.SEG_ID_A = 0
        self.SEG_ID_CLS = 2
        self.SEG_ID_PAD = 4
        self.CLS_ID = 3
        self.SEP_ID = 4
        self.training = tf.placeholder(tf.bool)
        self.num_gpus = num_gpus
        self.config_fname = config_fname
        self.max_seq_length = max_seq_length
        # define placeholders
        self.is_training = tf.placeholder(tf.bool)
        self.input_ids = tf.placeholder(tf.int32, [max_seq_length, batch_size * num_gpus], name='inputs_ids')
        self.input_ids_list = tf.split(self.input_ids, num_gpus, axis=1)
        self.input_mask = tf.placeholder(tf.float32, [max_seq_length, batch_size * num_gpus], name='input_mask')
        self.input_mask_list = tf.split(self.input_mask, num_gpus, axis=1)
        self.segment_ids = tf.placeholder(tf.int32, [max_seq_length, batch_size * num_gpus], name='segment_ids')
        self.segment_ids_list = tf.split(self.segment_ids, num_gpus, axis=1)
        self.label_ids = tf.placeholder(tf.int32, [batch_size * num_gpus], name='label_ids')
        self.label_ids_list = tf.split(self.label_ids, num_gpus, axis=0)
        # Load a corpus.
        super().__init__(train_corpus_fname=train_corpus_fname,
                         tokenized_train_corpus_fname=train_corpus_fname + ".xlnet-tokenized",
                         test_corpus_fname=test_corpus_fname, batch_size=batch_size * num_gpus,
                         tokenized_test_corpus_fname=test_corpus_fname + ".xlnet-tokenized",
                         model_name="xlnet", model_save_path=model_save_path,
                         sp_model_path=sp_model_path)
        self.train_steps = int(self.train_data_size * self.num_epochs / self.batch_size)
        self.eval_every = int(self.train_data_size / self.batch_size)  # epoch마다 평가

    def tune(self):
        with tf.device('/cpu:0'):
            global_step = tf.train.get_or_create_global_step()
            all_grads, all_vars, total_loss, total_logits, optimizers = [], [], [], [], []
            for k in range(self.num_gpus):
                with tf.device('/gpu:%d' % k), tf.variable_scope('tower%d' % k, reuse=tf.AUTO_REUSE):
                    optimizer = get_xlnet_optimizer(self.learning_rate, self.warmup_steps, self.decay_method,
                                                    self.adam_epsilon, self.train_steps, self.min_lr_ratio,
                                                    self.weight_decay, global_step)
                    logits, loss = make_xlnet_graph(self.input_ids_list[k], self.input_mask_list[k],
                                                    self.segment_ids_list[k], self.label_ids_list[k],
                                                    self.config_fname, self.num_labels,
                                                    self.is_training, tune=True)
                    grads_and_vars = optimizer.compute_gradients(loss)
                    gradients, variables = zip(*[el for el in grads_and_vars if el[0] is not None])
                    clipped, _ = tf.clip_by_global_norm(gradients, self.clip)
                    all_vars.append(variables)
                    all_grads.append(clipped)
                    optimizers.append(optimizer)
                    total_loss.append(loss)
                    total_logits.append(logits)
            self.logits = tf.concat(total_logits, axis=0)
            self.loss = tf.reduce_sum(total_loss)
            average_grads = allreduce_grads(all_grads, average=False)
            merged_grads_and_vars = merge_grad_list(average_grads, all_vars)
            train_ops = []
            for idx, grads_and_vars in enumerate(merged_grads_and_vars):
                with tf.name_scope('apply_gradients'), tf.device(tf.DeviceSpec(device_type="GPU", device_index=idx)):
                    train_ops.append(optimizers[idx].apply_gradients(grads_and_vars, global_step=global_step,
                                                                     name='apply_grad_{}'.format(idx)))
            train_op = tf.group(*train_ops, name='train_op')
            output_feed = [train_op, global_step, self.logits, self.loss]
            # Manually increment `global_step` for AdamWeightDecayOptimizer
            if isinstance(optimizers[0], AdamWeightDecayOptimizer):
                new_global_step = global_step + 1
                train_op = tf.group(train_op, [global_step.assign(new_global_step)])
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            load_pretrained_xlnet_model(self.pretrain_model_fname, self.num_gpus)
            # 0번 GPU의 param만 저장
            saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='tower0') + [global_step], max_to_keep=1)
            self.train(sess, saver, global_step, output_feed)

    def make_input(self, sentences, labels, is_training):
        features = {'input_ids': [], 'input_mask': [], 'segment_ids': []}
        for tokens in sentences:
            # classifier_utils의 convert_single_example 참고
            truncated_tokens = tokens[:(self.max_seq_length - 2)]
            input_ids = [self.tokenizer.PieceToId(token) for token in truncated_tokens] + [self.SEP_ID, self.CLS_ID]
            input_mask = [0] * len(input_ids)
            segment_ids = [self.SEG_ID_A] * (len(truncated_tokens) + 1) + [self.SEG_ID_CLS]
            if len(input_ids) < self.max_seq_length:
                delta_len = self.max_seq_length - len(input_ids)
                input_ids = [0] * delta_len + input_ids
                input_mask = [1] * delta_len + input_mask
                segment_ids = [self.SEG_ID_PAD] * delta_len + segment_ids
            features['input_ids'].append(input_ids)
            features['input_mask'].append(input_mask)
            features['segment_ids'].append(segment_ids)
        if is_training:
            input_feed = {
                self.is_training: is_training,
                self.input_ids: np.transpose(np.array(features['input_ids'])),
                self.segment_ids: np.transpose(np.array(features['segment_ids'])),
                self.input_mask: np.transpose(np.array(features['input_mask'])),
                self.label_ids: np.array(labels)
            }
        else:
            input_feed_ = {
                self.is_training: is_training,
                self.input_ids: np.transpose(np.array(features['input_ids'])),
                self.segment_ids: np.transpose(np.array(features['segment_ids'])),
                self.input_mask: np.transpose(np.array(features['input_mask'])),
                self.label_ids: np.array(labels)
            }
            input_feed = [input_feed_, labels]
        return input_feed


class WordEmbeddingTuner(Tuner):

    def __init__(self, train_corpus_fname, test_corpus_fname,
                 model_save_path, embedding_name, embedding_fname=None,
                 embedding_size=100, batch_size=128, learning_rate=0.0001, num_labels=2):
        # Load a corpus.
        super().__init__(train_corpus_fname=train_corpus_fname,
                         tokenized_train_corpus_fname=train_corpus_fname + ".word-embedding-tokenized",
                         test_corpus_fname=test_corpus_fname, batch_size=batch_size,
                         tokenized_test_corpus_fname=test_corpus_fname + ".word-embedding-tokenized",
                         model_name=embedding_name, model_save_path=model_save_path)
        self.lr = learning_rate
        self.embedding_size = embedding_size
        # Load Pretrained Word Embeddings.
        self.embeddings, self.vocab = self.load_embeddings(embedding_name, embedding_fname)
        self.unk_idx = len(self.vocab)
        self.pad_idx = len(self.vocab) + 1
        # build train graph.
        self.ids_placeholder, self.input_lengths, self.labels_placeholder, \
        self.dropout_keep_prob, self.embedding_placeholder, self.embed_init, \
        self.logits, self.loss = make_word_embedding_graph(num_labels, len(self.vocab) + 2, self.embedding_size, tune=True)

    def tune(self):
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        output_feed = [train_op, global_step, self.logits, self.loss]
        saver = tf.train.Saver(max_to_keep=1)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(self.embed_init, feed_dict={self.embedding_placeholder: self.embeddings})
        self.train(sess, saver, global_step, output_feed)

    def make_input(self, sentences, labels, is_training):
        input_ids, lengths = [], []
        max_token_length = self.get_max_token_length_this_batch(sentences)
        for tokens in sentences:
            token_ids = []
            tokens_length = len(tokens)
            for token in tokens:
                if token in self.vocab:
                    token_ids.append(self.vocab[token])
                else:
                    token_ids.append(self.unk_idx)
            if len(tokens) < max_token_length:
                token_ids.extend(
                    [self.pad_idx] * (max_token_length - tokens_length))
            input_ids.append(token_ids)
            lengths.append(len(token_ids))
        if is_training:
            input_feed = {
                self.ids_placeholder: np.array(input_ids),
                self.input_lengths: np.array(lengths),
                self.labels_placeholder: np.array(labels),
                self.dropout_keep_prob: 0.9
            }
        else:
            input_feed = {
                self.ids_placeholder: np.array(input_ids),
                self.input_lengths: np.array(lengths),
                self.labels_placeholder: np.array(labels),
                self.dropout_keep_prob: 1.0
            }
            input_feed = [input_feed, labels]
        return input_feed

    def get_max_token_length_this_batch(self, sentences):
        return max(len(sentence) for sentence in sentences)

    def get_truncated_normal(self, mean=0, sd=1, low=-1, upp=1):
        return truncnorm(
            (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

    def load_embeddings(self, embedding_name, embedding_fname):
        random_generator = self.get_truncated_normal()
        if embedding_name in ["fasttext", "glove", "swivel"]:
            embeddings, words = [], []
            with open(embedding_fname, 'r') as f:
                if embedding_name == "fasttext":
                    next(f) # skip head line
                for line in f:
                    if embedding_name == "swivel":
                        splitedLine = line.strip().split("\t")
                    else:
                        splitedLine = line.strip().split()
                    word = splitedLine[0]
                    embedding = [float(el) for el in splitedLine[1:]]
                    words.append(word)
                    embeddings.append(embedding)
            embeddings = np.array(embeddings)
            vocab = {word:idx for idx, word in enumerate(words)}
        elif embedding_name == "word2vec":
            model = Word2Vec.load(embedding_fname)
            embeddings = model.wv.vectors
            vocab = {word:idx for idx, word in enumerate(model.wv.index2word)}
        else:
            words_count = defaultdict(int)
            for tokens, _ in self.train_data:
                for token in tokens:
                    words_count[token] += 1
            sorted_words = sorted(words_count.items(), key=lambda x: x[1], reverse=True)[:50000]
            words = [word for word, _ in sorted_words]
            vocab = {word:idx for idx, word in enumerate(words)}
            random_embeddings = random_generator.rvs(len(vocab) * self.embedding_size)
            embeddings = random_embeddings.reshape(len(vocab), self.embedding_size)
        # for UNK, PAD token
        added_embeddings = random_generator.rvs(self.embedding_size * 2)
        embeddings = np.append(embeddings, added_embeddings.reshape(2, self.embedding_size), axis=0)
        return embeddings, vocab


def get_xlnet_optimizer(learning_rate, warmup_steps, decay_method, adam_epsilon,
                        train_steps, min_lr_ratio, weight_decay, global_step):
    # increase the learning rate linearly
    if warmup_steps > 0:
        warmup_lr = (tf.cast(global_step, tf.float32)
                     / tf.cast(warmup_steps, tf.float32)
                     * learning_rate)
    else:
        warmup_lr = 0.0
        # decay the learning rate
    if decay_method == "poly":
        decay_lr = tf.train.polynomial_decay(
            learning_rate,
            global_step=global_step - warmup_steps,
            decay_steps=train_steps - warmup_steps,
            end_learning_rate=learning_rate * min_lr_ratio)
    elif decay_method == "cos":
        decay_lr = tf.train.cosine_decay(
            learning_rate,
            global_step=global_step - warmup_steps,
            decay_steps=train_steps - warmup_steps,
            alpha=min_lr_ratio)
    else:
        raise ValueError(decay_method)
    learning_rate = tf.where(global_step < warmup_steps, warmup_lr, decay_lr)
    if weight_decay == 0:
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            epsilon=adam_epsilon)
    else:
        optimizer = AdamWeightDecayOptimizer(
            learning_rate=learning_rate,
            epsilon=adam_epsilon,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
            weight_decay_rate=weight_decay)
    return optimizer


def allreduce_grads(all_grads, average):
    """
    All-reduce average the gradients among K devices. Results are broadcasted to all devices.
    Args:
        all_grads (K x N): List of list of gradients. N is the number of variables.
        average (bool): average gradients or not.
    Returns:
        K x N: same as input, but each grad is replaced by the average over K devices.
    """
    nr_tower = len(all_grads)
    if nr_tower == 1:
        return all_grads
    new_all_grads = []  # N x K
    for grads in zip(*all_grads):
        summed = nccl.all_sum(grads)
        grads_for_devices = []  # K
        for g in summed:
            with tf.device(g.device):
                # tensorflow/benchmarks didn't average gradients
                if average:
                    g = tf.multiply(g, 1.0 / nr_tower)
            grads_for_devices.append(g)
        new_all_grads.append(grads_for_devices)
    # transpose to K x N
    ret = list(zip(*new_all_grads))
    return ret


def merge_grad_list(all_grads, all_vars):
    """
    Args:
        all_grads (K x N): gradients
        all_vars(K x N): variables
    Return:
        K x N x 2: list of list of (grad, var) pairs
    """
    return [list(zip(gs, vs)) for gs, vs in zip(all_grads, all_vars)]


def load_pretrained_xlnet_model(pretrained_model_fname, num_gpus):
    tf.logging.info("Initialize from the ckpt {}".format(pretrained_model_fname))
    name_to_variable = collections.OrderedDict()
    for var in tf.trainable_variables():
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var
    init_vars = tf.train.list_variables(pretrained_model_fname)
    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        for k in range(num_gpus):
            assignment_map[name] = name_to_variable['tower' + str(k) + '/' + name]
    tf.train.init_from_checkpoint(pretrained_model_fname, assignment_map)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='model name')
    parser.add_argument('--train_corpus_fname', type=str, help='train corpus file name')
    parser.add_argument('--test_corpus_fname', type=str, help='test corpus file name')
    parser.add_argument('--vocab_fname', type=str, help='vocab file name')
    parser.add_argument('--pretrain_model_fname', type=str, help='pretrained model file name')
    parser.add_argument('--config_fname', type=str, help='config file name')
    parser.add_argument('--model_save_path', type=str, help='model save path')
    parser.add_argument('--embedding_name', type=str, help='embedding name')
    parser.add_argument('--embedding_fname', type=str, help='embedding file path')
    parser.add_argument('--num_gpus', type=str, help='number of GPUs (XLNet only)')
    args = parser.parse_args()

    if args.model_name == "elmo":
        model = ELMoTuner(train_corpus_fname=args.train_corpus_fname,
                          test_corpus_fname=args.test_corpus_fname,
                          vocab_fname=args.vocab_fname,
                          options_fname=args.config_fname,
                          pretrain_model_fname=args.pretrain_model_fname,
                          model_save_path=args.model_save_path)
    elif args.model_name == "bert":
        model = BERTTuner(train_corpus_fname=args.train_corpus_fname,
                          test_corpus_fname=args.test_corpus_fname,
                          vocab_fname=args.vocab_fname,
                          pretrain_model_fname=args.pretrain_model_fname,
                          bertconfig_fname=args.config_fname,
                          model_save_path=args.model_save_path)
    elif args.model_name == "xlnet":
        if args.num_gpus is None:
            num_gpus = 1
        else:
            num_gpus = int(args.num_gpus)
        model = XLNetTuner(train_corpus_fname=args.train_corpus_fname,
                           test_corpus_fname=args.test_corpus_fname,
                           pretrain_model_fname=args.pretrain_model_fname,
                           config_fname=args.config_fname,
                           model_save_path=args.model_save_path,
                           sp_model_path=args.vocab_fname,
                           num_gpus=num_gpus)
    elif args.model_name == "word":
        model = WordEmbeddingTuner(train_corpus_fname=args.train_corpus_fname,
                                   test_corpus_fname=args.test_corpus_fname,
                                   embedding_name=args.embedding_name,
                                   embedding_fname=args.embedding_fname,
                                   model_save_path=args.model_save_path)
    else:
        model = None
    model.tune()