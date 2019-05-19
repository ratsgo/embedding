import sys, os, random
import numpy as np
import tensorflow as tf
from pytorch_pretrained_bert.tokenization import BertTokenizer
from preprocess import get_tokenizer, post_processing

sys.path.append('models')
from bilm import Batcher, BidirectionalLanguageModel, weight_layers
from bert.modeling import BertModel, BertConfig
from bert.optimization import create_optimizer


class Tuner(object):

    def __init__(self, train_corpus_fname=None, tokenized_train_corpus_fname=None,
                 test_corpus_fname=None, tokenized_test_corpus_fname=None,
                 model_name="bert", model_save_path=None, vocab_fname=None, eval_every=1000,
                 batch_size=32, num_epochs=3, dropout_keep_prob_rate=0.9, model_ckpt_path=None):
        # configurations
        self.model_name = model_name
        self.eval_every = eval_every
        self.model_ckpt_path = model_ckpt_path
        self.model_save_path = model_save_path
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.dropout_keep_prob_rate = dropout_keep_prob_rate
        self.best_valid_score = 0.0
        # define tokenizer
        if self.model_name == "bert":
            self.tokenizer = BertTokenizer(vocab_file=vocab_fname, do_lower_case=False)
        else:
            self.tokenizer = get_tokenizer("mecab")
        # load or tokenize corpus
        self.train_data, self.train_data_size = self.load_or_tokenize_corpus(train_corpus_fname, tokenized_train_corpus_fname)
        self.test_data, self.test_data_size = self.load_or_tokenize_corpus(test_corpus_fname, tokenized_test_corpus_fname)

    def load_or_tokenize_corpus(self, corpus_fname, tokenized_corpus_fname):
        data_set = []
        if os.path.exists(tokenized_corpus_fname):
            print("load tokenized corpus :", tokenized_corpus_fname)
            with open(tokenized_corpus_fname, 'r') as f1:
                for line in f1:
                    tokens, label = line.strip().split("\u241E")
                    if len(tokens) > 0:
                        data_set.append([tokens.split(" "), int(label)])
        else:
            print("tokenize corpus :", corpus_fname, ">", tokenized_corpus_fname)
            with open(corpus_fname, 'r') as f2:
                next(f2)  # skip head line
                for line in f2:
                    _, sentence, label = line.strip().split("\t")
                    if self.model_name == "bert":
                        tokens = self.tokenizer.tokenize(sentence)
                    else:
                        tokens = self.tokenizer.morphs(sentence)
                        tokens = post_processing(tokens)
                    if int(label) > 2.5:
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
        for batch in test_batches:
            current_input_feed, current_labels = batch
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
        num_batches_per_epoch = int((data_size - 1) / self.batch_size) + 1
        if is_training:
            tf.logging.info("num_batches_per_epoch : " + str(num_batches_per_epoch))
        for epoch in range(num_epochs):
            idx = random.sample(range(data_size), data_size)
            data = np.array(data)[idx]
            for batch_num in range(num_batches_per_epoch):
                batch_sentences = []
                batch_labels = []
                start_index = batch_num * self.batch_size
                end_index = min((batch_num + 1) * self.batch_size, data_size)
                features = data[start_index:end_index]
                for feature in features:
                    sentence, label = feature
                    batch_sentences.append(sentence)
                    batch_labels.append(int(label))
                yield self.make_input(batch_sentences, batch_labels, is_training)

    def buildGraph(self):
        raise NotImplementedError

    def make_input(self, sentences, labels, is_training):
        raise NotImplementedError

    def tune(self):
        raise NotImplementedError


class ELMoTuner(Tuner):

    def __init__(self, train_corpus_fname, test_corpus_fname,
                 vocab_fname, options_fname, pretrain_model_fname,
                 model_save_path, max_characters_per_token=30, batch_size=32):
        # Load a corpus.
        super().__init__(train_corpus_fname=train_corpus_fname,
                         tokenized_train_corpus_fname=train_corpus_fname + ".elmo.tokenized",
                         test_corpus_fname=test_corpus_fname,
                         tokenized_test_corpus_fname=test_corpus_fname + ".elmo.tokenized",
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
        self.ids_placeholder, self.labels_placeholder, self.dropout_keep_prob, self.logits, self.loss = self.buildGraph()

    def buildGraph(self):
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
        bilm = BidirectionalLanguageModel(self.options_fname, self.pretrain_model_fname)
        # Input placeholders to the biLM.
        ids_placeholder = tf.placeholder(tf.int32, shape=(None, None, self.max_characters_per_token), name='input')
        # Output placeholders to the fine-tuned Net.
        labels_placeholder = tf.placeholder(tf.int32, shape=(None))
        # Get ops to compute the LM embeddings.
        embeddings_op = bilm(ids_placeholder)
        # Get lengths.
        input_lengths = embeddings_op['lengths']
        # define dropout & train
        dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
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
        # (batch_size, seq_len, HIDDEN_SIZE)
        H = tf.nn.tanh(output_fw + output_bw)
        # softmax(dot(W, H)) : (batch_size, seq_len, 1)
        attention_score = tf.nn.softmax(tf.contrib.layers.fully_connected(inputs=H, num_outputs=1, activation_fn=None))
        # dot(prob, H) : (batch_size, HIDDEN_SIZE, 1) > (batch_size, HIDDEN_SIZE)
        attention_output = tf.squeeze(tf.matmul(tf.transpose(H, perm=[0, 2, 1]), attention_score), axis=-1)
        layer_output = tf.nn.dropout(tf.nn.tanh(attention_output), dropout_keep_prob)

        # Feed-Forward Layer
        fc = tf.contrib.layers.fully_connected(inputs=layer_output,
                                               num_outputs=512,
                                               activation_fn=tf.nn.relu,
                                               weights_initializer=tf.contrib.layers.xavier_initializer(),
                                               biases_initializer=tf.zeros_initializer())
        features_drop = tf.nn.dropout(fc, dropout_keep_prob)
        logits = tf.contrib.layers.fully_connected(inputs=features_drop,
                                                   num_outputs=self.num_labels,
                                                   activation_fn=None,
                                                   weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                   biases_initializer=tf.zeros_initializer())
        # Loss Layer
        CE = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=logits)
        loss = tf.reduce_mean(CE)
        return ids_placeholder, labels_placeholder, dropout_keep_prob, logits, loss

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
                 max_seq_length=32, warmup_proportion=0.1,
                 batch_size=32, learning_rate=5e-5):
        # Load a corpus.
        super().__init__(train_corpus_fname=train_corpus_fname,
                         tokenized_train_corpus_fname=train_corpus_fname + ".bert.tokenized",
                         test_corpus_fname=test_corpus_fname, batch_size=batch_size,
                         tokenized_test_corpus_fname=test_corpus_fname + ".bert.tokenized",
                         model_name="bert", vocab_fname=vocab_fname, model_save_path=model_save_path)
        # configurations
        self.config = BertConfig.from_json_file(bertconfig_fname)
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
        self.input_ids, self.input_mask, self.segment_ids, self.label_ids, self.logits, self.loss = self.buildGraph()

    def buildGraph(self):
        # define input placeholders
        input_ids = tf.placeholder(tf.int32, [None, self.max_seq_length], name='inputs_ids')
        input_mask = tf.placeholder(tf.int32, [None, self.max_seq_length], name='input_mask')
        segment_ids = tf.placeholder(tf.int32, [None, self.max_seq_length], name='segment_ids')
        label_ids = tf.placeholder(tf.int32, [None], name='label_ids')
        # build BERT model
        model = BertModel(config=self.config,
                          is_training=True,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          token_type_ids=segment_ids)
        # build tune layer
        bert_embeddings_dropout = tf.nn.dropout(model.pooled_output, keep_prob=(1 - self.dropout_keep_prob_rate))
        logits = tf.contrib.layers.fully_connected(inputs=bert_embeddings_dropout,
                                                   num_outputs=self.num_labels,
                                                   activation_fn=None,
                                                   weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                   biases_initializer=tf.zeros_initializer())
        # loss layer
        CE = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_ids, logits=logits)
        loss = tf.reduce_mean(CE)
        return input_ids, input_mask, segment_ids, label_ids, logits, loss

    def tune(self):
        global_step = tf.train.get_or_create_global_step()
        tf.logging.info("num_train_steps: " + str(self.num_train_steps))
        tf.logging.info("num_warmup_steps: " + str(self.num_warmup_steps))
        train_op = create_optimizer(self.loss, self.learning_rate, self.num_train_steps, self.num_warmup_steps, use_tpu=False)
        output_feed = [train_op, global_step, self.logits, self.loss]
        restore_vars = [v for v in tf.trainable_variables() if "bert" in v.name]
        sess = tf.Session()
        tf.train.Saver(restore_vars).restore(sess, self.pretrain_model_fname)
        saver = tf.train.Saver(max_to_keep=1)
        sess.run(tf.global_variables_initializer())
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


if __name__ == '__main__':
    model_name = sys.argv[1]
    train_corpus_fname = sys.argv[2]
    test_corpus_fname = sys.argv[3]
    vocab_fname = sys.argv[4]
    pretrain_model_fname = sys.argv[5]
    config_fname = sys.argv[6]
    model_save_path = sys.argv[7]
    if model_name == "elmo":
        model = ELMoTuner(train_corpus_fname=train_corpus_fname,
                          test_corpus_fname=test_corpus_fname,
                          vocab_fname=vocab_fname,
                          options_fname=config_fname,
                          pretrain_model_fname=pretrain_model_fname,
                          model_save_path=model_save_path)
    elif model_name == "bert":
        model = BERTTuner(train_corpus_fname=train_corpus_fname,
                          test_corpus_fname=test_corpus_fname,
                          vocab_fname=vocab_fname,
                          pretrain_model_fname=pretrain_model_fname,
                          bertconfig_fname=config_fname,
                          model_save_path=model_save_path)
    model.tune()