import tensorflow as tf
from tensorflow.contrib import rnn, seq2seq, layers
from tensorflow.contrib.framework import nest

END = 1

class Seq2Seq:
    def __init__(self, vocab_size, batch_size=16, learning_rate=0.001,
                 embed_dim=100, beam_width=10,
                 cell_type=rnn.LSTMCell, hidden_size=512, depth=2,
                 residual=True, dropout=True):
        self.cell_type = cell_type
        self.hidden_size = hidden_size
        self.depth = depth
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.beam_width = beam_width
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.residual = residual
        self.dropout = dropout
        if dropout:
            self.keep_prob = tf.placeholder(tf.float32, shape=None)

        self.sequence_length = tf.placeholder(tf.int32, shape=(None,)) # TODO
        self.X = tf.placeholder(tf.int32, shape=(None, None))
        self.y = tf.placeholder(tf.int32, shape=(None, None))

        # Assemble the model graph
        encoder_outputs, encoder_final_state = self._make_encoder()
        decoder_cell, decoder_initial_state = self._make_decoder(encoder_outputs, encoder_final_state)
        self.loss_op, self.train_op = self._make_train(decoder_cell, decoder_initial_state)
        tvars = tf.trainable_variables()
        decoder_cell, decoder_initial_state = self._make_decoder(encoder_outputs, encoder_final_state, beam_search=True, reuse=True)
        self.pred_op = self._make_predict(decoder_cell, decoder_initial_state)

        # DEBUG: Checking that variables are properly reused
        tvars_after = tf.trainable_variables()
        for v in tvars_after:
            if v not in tvars:
                print(v)

    def _make_cell(self, hidden_size=None):
        cell = self.cell_type(hidden_size or self.hidden_size)
        if self.dropout:
            cell = rnn.DropoutWrapper(cell, self.keep_prob)
        if self.residual:
            cell = rnn.ResidualWrapper(cell)
        return cell

    def _make_encoder(self):
        inputs = layers.embed_sequence(
            self.X,
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            scope='embed')

        # Project to correct dimensions
        inputs = tf.layers.dense(inputs, self.hidden_size//2)

        cell_fw = rnn.MultiRNNCell([
            self._make_cell(self.hidden_size//2) for _ in range(self.depth)
        ])
        cell_bw = rnn.MultiRNNCell([
            self._make_cell(self.hidden_size//2) for _ in range(self.depth)
        ])
        encoder_outputs, encoder_final_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw, cell_bw=cell_bw, sequence_length=self.sequence_length,
            inputs=inputs, dtype=tf.float32)

        # Concat forward and backward outputs
        encoder_outputs = tf.concat(encoder_outputs, 2)

        # Concat forward and backward states
        encoder_fw_states, encoder_bw_states = encoder_final_state
        encoder_final_state = []
        for fw, bw in zip(encoder_fw_states, encoder_bw_states):
            c = tf.concat([fw.c, bw.c], 1)
            h = tf.concat([fw.h, bw.h], 1)
            encoder_final_state.append(rnn.LSTMStateTuple(c=c, h=h))
        return encoder_outputs, encoder_final_state

    def _make_decoder(self, encoder_outputs, encoder_final_state, beam_search=False, reuse=False):
        with tf.variable_scope('decode', reuse=reuse):
            cells = [self._make_cell() for _ in range(self.depth)]

            if beam_search:
                encoder_outputs = seq2seq.tile_batch(
                    encoder_outputs, multiplier=self.beam_width)
                encoder_final_state = nest.map_structure(
                    lambda s: seq2seq.tile_batch(s, multiplier=self.beam_width),
                    encoder_final_state)
                sequence_length = seq2seq.tile_batch(
                    self.sequence_length, multiplier=self.beam_width)
            else:
                sequence_length = self.sequence_length

            # Prepare attention mechanism;
            # add only to last cell
            attention_mechanism = seq2seq.LuongAttention(
                num_units=self.hidden_size, memory=encoder_outputs,
                memory_sequence_length=sequence_length, name='attn')
            cells[-1] = seq2seq.AttentionWrapper(
                cells[-1], attention_mechanism, attention_layer_size=self.hidden_size,
                initial_cell_state=encoder_final_state[-1],
                cell_input_fn=lambda inp, attn: tf.layers.dense(tf.concat([inp, attn], -1), self.hidden_size),
                name='attnwrap'
            )

            # Copy encoder final state as decoder initial state
            decoder_initial_state = [s for s in encoder_final_state]

            batch_size = self.batch_size
            if beam_search: batch_size *= self.beam_width
            decoder_initial_state[-1] = cells[-1].zero_state(
                dtype=tf.float32, batch_size=batch_size)

            cell = rnn.MultiRNNCell(cells)
            return cell, tuple(decoder_initial_state)

    def _make_train(self, decoder_cell, decoder_initial_state):
        # Assume 0 is the START token
        start_tokens = tf.zeros((self.batch_size,), dtype=tf.int32)
        y = tf.concat([tf.expand_dims(start_tokens, 1), self.y], 1)
        output_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(y, 1)), 1)

        # Reuse encoding embeddings
        inputs = layers.embed_sequence(
            y,
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            scope='embed', reuse=True)

        # Project to correct dimensions
        # proj = tf.layers.Dense(self.vocab_size, name='output_projection')

        # Prepare the decoder with the attention cell
        with tf.variable_scope('decode'):
            inputs = tf.layers.dense(inputs, self.hidden_size, name='input_proj')
            helper = seq2seq.TrainingHelper(inputs, output_lengths)
            decoder = seq2seq.BasicDecoder(
                cell=decoder_cell, helper=helper,
                initial_state=decoder_initial_state)
                # output_layer=proj)
            final_outputs, final_state, final_sequence_lengths = seq2seq.dynamic_decode(
                decoder=decoder, impute_finished=True)

        # Prioritize examples that the model was wrong on,
        # by setting weight=1 to any example where the prediction was not 1,
        # i.e. incorrect
        weights = tf.to_float(tf.not_equal(y[:, :-1], 1))

        # Training and loss ops
        loss_op = seq2seq.sequence_loss(
            final_outputs.rnn_output, self.y, weights=weights)
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss_op)
        return loss_op, train_op

    def _make_predict(self, decoder_cell, decoder_initial_state):
        # Access embeddings directly
        with tf.variable_scope('embed', reuse=True):
            embeddings = tf.get_variable('embeddings')

        start_tokens = tf.zeros((self.batch_size,), dtype=tf.int32)

        with tf.variable_scope('decode', reuse=True):
            embeddings = tf.layers.dense(embeddings, self.hidden_size, name='input_proj')
            decoder = seq2seq.BeamSearchDecoder(
                cell=decoder_cell,
                embedding=embeddings,
                start_tokens=start_tokens,
                end_token=END,
                initial_state=decoder_initial_state,
                beam_width=self.beam_width
            )

            final_outputs, final_state, final_sequence_lengths = seq2seq.dynamic_decode(
                decoder=decoder, impute_finished=False)
        return final_outputs
