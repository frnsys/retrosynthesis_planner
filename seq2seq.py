import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn, seq2seq, layers
from tensorflow.contrib.framework import nest

START = 0
END = 1


def pad_arrays(arrs, pad_value=END):
    """Pad list of ragged arrays with `pad_value` so
    they all have the same shape"""
    max_len = np.max([len(a) for a in arrs])
    return np.asarray([np.pad(a, (0, max_len - len(a)),
                              'constant', constant_values=pad_value) for a in arrs])


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

        self.X = tf.placeholder(tf.int32, shape=(None, None))
        self.y = tf.placeholder(tf.int32, shape=(None, None))
        self.sequence_length = tf.reduce_sum(tf.to_int32(tf.not_equal(self.X, END)), 1)

        # Assemble the model graph
        encoder_outputs, encoder_final_state = self._make_encoder()
        decoder_cell, decoder_initial_state = self._make_decoder(encoder_outputs, encoder_final_state)
        self.loss_op, self.train_op, self.acc_op = self._make_train(decoder_cell, decoder_initial_state)
        tvars = tf.trainable_variables()
        decoder_cell, decoder_initial_state = self._make_decoder(encoder_outputs, encoder_final_state, beam_search=True, reuse=True)
        self.pred_op = self._make_predict(decoder_cell, decoder_initial_state)

        # DEBUG: Checking that variables are properly reused
        tvars_after = tf.trainable_variables()
        for v in tvars_after:
            if v not in tvars:
                print(v)

    def _make_cell(self, hidden_size=None):
        """Create a single RNN cell"""
        cell = self.cell_type(hidden_size or self.hidden_size)
        if self.dropout:
            cell = rnn.DropoutWrapper(cell, self.keep_prob)
        if self.residual:
            cell = rnn.ResidualWrapper(cell)
        return cell

    def _make_encoder(self):
        """Create the encoder"""
        inputs = layers.embed_sequence(
            self.X,
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            scope='embed')

        # Project to correct dimensions
        # Halve the dimensions so that
        # the bidirectional output has the correct size
        # (because we concat the forward and backward outputs,
        # the output size is 2*size)
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

        # Concat forward and backward layer states
        encoder_fw_states, encoder_bw_states = encoder_final_state
        encoder_final_state = []
        for fw, bw in zip(encoder_fw_states, encoder_bw_states):
            c = tf.concat([fw.c, bw.c], 1)
            h = tf.concat([fw.h, bw.h], 1)
            encoder_final_state.append(rnn.LSTMStateTuple(c=c, h=h))
        return encoder_outputs, encoder_final_state

    def _make_decoder(self, encoder_outputs, encoder_final_state, beam_search=False, reuse=False):
        """Create decoder"""
        with tf.variable_scope('decode', reuse=reuse):
            # Create decoder cells
            cells = [self._make_cell() for _ in range(self.depth)]

            if beam_search:
                # Tile inputs as needed for beam search
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

            # Set last initial state to be AttentionWrapperState
            batch_size = self.batch_size
            if beam_search: batch_size *= self.beam_width
            decoder_initial_state[-1] = cells[-1].zero_state(
                dtype=tf.float32, batch_size=batch_size)

            # Wrap up the cells
            cell = rnn.MultiRNNCell(cells)

            # Return initial state as a tuple
            # (required by tensorflow)
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

        # Prepare the decoder with the attention cell
        with tf.variable_scope('decode'):
            # Project to correct dimensions
            out_proj = tf.layers.Dense(self.vocab_size, name='output_proj')
            inputs = tf.layers.dense(inputs, self.hidden_size, name='input_proj')

            helper = seq2seq.TrainingHelper(inputs, output_lengths)
            decoder = seq2seq.BasicDecoder(
                cell=decoder_cell, helper=helper,
                initial_state=decoder_initial_state,
                output_layer=out_proj)
            final_outputs, final_state, final_sequence_lengths = seq2seq.dynamic_decode(
                decoder=decoder, impute_finished=True)
            logits = final_outputs.rnn_output

        # Prioritize examples that the model was wrong on,
        # by setting weight=1 to any example where the prediction was not 1,
        # i.e. incorrect
        weights = tf.to_float(tf.not_equal(y[:, :-1], 1))

        # Training and loss ops
        loss_op = seq2seq.sequence_loss(logits, self.y, weights=weights)
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss_op)
        pred_idx = tf.to_int32(tf.argmax(logits, 2))
        accuracy_op = tf.reduce_mean(tf.cast(tf.equal(pred_idx, self.y), tf.float32), name='acc')
        return loss_op, train_op, accuracy_op

    def _make_predict(self, decoder_cell, decoder_initial_state):
        # Access embeddings directly
        with tf.variable_scope('embed', reuse=True):
            embeddings = tf.get_variable('embeddings')

        # Assume 0 is the START token
        start_tokens = tf.zeros((self.batch_size,), dtype=tf.int32)

        # For predictions, we use beam search to return multiple results
        with tf.variable_scope('decode', reuse=True):
            # Project to correct dimensions
            out_proj = tf.layers.Dense(self.vocab_size, name='output_proj')
            embeddings = tf.layers.dense(embeddings, self.hidden_size, name='input_proj')

            decoder = seq2seq.BeamSearchDecoder(
                cell=decoder_cell,
                embedding=embeddings,
                start_tokens=start_tokens,
                end_token=END,
                initial_state=decoder_initial_state,
                beam_width=self.beam_width,
                output_layer=out_proj
            )

            final_outputs, final_state, final_sequence_lengths = seq2seq.dynamic_decode(
                decoder=decoder, impute_finished=False)
        return final_outputs

if __name__ == '__main__':
    import os
    import random
    from tqdm import tqdm, trange

    # Load and save vocab
    vocab = ['<S>', '</S>']
    with open('data/vocab.dat', 'r') as f:
        for line in tqdm(f, desc='vocab'):
            tok, _ = line.strip().split('\t')
            vocab.append(tok)
    vocab2id = {v: i for i, v in enumerate(vocab)}
    with open('data/vocab.idx', 'w') as f:
        f.write('\n'.join(vocab))

    # Load reactions
    X, y = [], []
    with open('data/reactions.dat', 'r') as f:
        for line in tqdm(f, desc='reactions'):
            source_toks, target_toks = line.strip().split('\t')
            source_toks = source_toks.split()
            target_toks = target_toks.split()
            source_doc = [vocab2id['<S>']] + [vocab2id[tok] for tok in source_toks] + [END]
            target_doc = [vocab2id['<S>']] + [vocab2id[tok] for tok in target_toks] + [END]

            # Since this is for retrosynthesis,
            # we want predict reactants (sources) from products (targets)
            X.append(target_doc)
            y.append(source_doc)

    print('Preparing model...')
    batch_size = 16
    # model = Seq2Seq(vocab_size=len(vocab),
    #                 batch_size=batch_size, learning_rate=0.0001,
    #                 embed_dim=100, hidden_size=128, depth=1,
    #                 beam_width=10, residual=True, dropout=True)
    model = Seq2Seq(vocab_size=len(vocab),
                    batch_size=batch_size, learning_rate=0.0001,
                    embed_dim=256, hidden_size=1024, depth=2,
                    beam_width=10, residual=True, dropout=True)


    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    save_path = 'model'
    ckpt_path = os.path.join(save_path, 'model.ckpt')
    saver = tf.train.Saver()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        saver.restore(sess, ckpt_path)

    print('Training...')
    losses = []
    accuracy = []
    epochs = 3
    it = trange(epochs)
    n_steps = int(np.ceil(len(X)/batch_size))
    for e in it:
        # Shuffle
        # For numpy arrays:
        # p = np.random.permutation(len(X))
        # X, y = X[p], y[p]
        # For python lists:
        xy = list(zip(X, y))
        random.shuffle(xy)
        X, y = zip(*xy)

        # Iterate batches
        eit = trange(n_steps)
        for i in eit:
            l = i*batch_size
            u = l + batch_size
            X_batch, y_batch = pad_arrays(X[l:u]), pad_arrays(y[l:u])
            _, err, acc = sess.run(
                [model.train_op, model.loss_op, model.acc_op],
                feed_dict={
                    model.keep_prob: 0.6,
                    model.X: X_batch,
                    model.y: y_batch
                }
            )
            losses.append(err)
            accuracy.append(acc)
            eit.set_postfix(
                loss=err,
                acc=acc,
                u_loss=np.mean(losses[-10:]) if losses else None,
                u_acc=np.mean(accuracy[-10:]) if accuracy else None)
        it.set_postfix(
            loss=np.mean(losses[-10:]),
            acc=np.mean(accuracy[-10:]))

        saver.save(sess, ckpt_path)
