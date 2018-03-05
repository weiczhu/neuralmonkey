import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.dataset import Dataset
from neuralmonkey.model.model_part import ModelPart, FeedDict, InitializerSpecs
from neuralmonkey.model.stateful import TemporalStateful
from neuralmonkey.vocabulary import Vocabulary
from neuralmonkey.decorators import tensor
from neuralmonkey.tf_utils import get_variable


class SequenceLabeler(ModelPart):
    """Classifier assing a label to each encoder's state."""

    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 input_sequence: TemporalStateful,
                 vocabulary: Vocabulary,
                 data_id: str,
                 dropout_keep_prob: float = 1.0,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None) -> None:
        check_argument_types()
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint,
                           initializers)

        self.input_sequence = input_sequence
        self.vocabulary = vocabulary
        self.data_id = data_id
        self.dropout_keep_prob = dropout_keep_prob

        self.input_size = int(
            self.input_sequence.temporal_states.get_shape()[-1])

        with self.use_scope():
            self.train_targets = tf.placeholder(
                tf.int32, [None, None], "labeler_targets")
            self.train_weights = tf.placeholder(
                tf.float32, [None, None], "labeler_padding_weights")
    # pylint: enable=too-many-arguments

    @tensor
    def decoding_w(self) -> tf.Variable:
        return get_variable(
            name="state_to_word_W",
            shape=[self.input_size, len(self.vocabulary)],
            initializer=tf.glorot_normal_initializer())

    @tensor
    def decoding_b(self) -> tf.Variable:
        return get_variable(
            name="state_to_word_b",
            shape=[len(self.vocabulary)],
            initializer=tf.zeros_initializer())

    @tensor
    def decoding_residual_w(self) -> tf.Variable:
        input_dim = self.input_sequence.input_sequence.dimension
        return get_variable(
            name="emb_to_word_W",
            shape=[input_dim, len(self.vocabulary)],
            initializer=tf.glorot_normal_initializer())

    @tensor
    def logits(self) -> tf.Tensor:
        # To multiply 3-D matrix (encoder hidden states) by a 2-D matrix
        # (weights), we use 1-by-1 convolution (similar trick can be found in
        # attention computation)

        # TODO dropout needs to be revisited

        intpus_states = tf.expand_dims(self.input_sequence.temporal_states, 2)
        weights_4d = tf.expand_dims(tf.expand_dims(self.decoding_w, 0), 0)

        multiplication = tf.nn.conv2d(
            intpus_states, weights_4d, [1, 1, 1, 1], "SAME")
        multiplication_3d = tf.squeeze(multiplication, squeeze_dims=[2])

        biases_3d = tf.expand_dims(tf.expand_dims(self.decoding_b, 0), 0)
        logits = multiplication_3d + biases_3d

        if hasattr(self.input_sequence, "input_sequence"):
            embedded_inputs = tf.expand_dims(
                self.input_sequence.input_sequence.temporal_states, 2)
            dweights_4d = tf.expand_dims(
                tf.expand_dims(self.decoding_residual_w, 0), 0)

            dmultiplication = tf.nn.conv2d(
                embedded_inputs, dweights_4d, [1, 1, 1, 1], "SAME")
            dmultiplication_3d = tf.squeeze(dmultiplication, squeeze_dims=[2])

            logits += dmultiplication_3d
        return logits

    @tensor
    def logprobs(self) -> tf.Tensor:
        return tf.nn.log_softmax(self.logits)

    @tensor
    def decoded(self) -> tf.Tensor:
        return tf.argmax(self.logits, 2)

    @tensor
    def cost(self) -> tf.Tensor:
        min_time = tf.minimum(tf.shape(self.train_targets)[1],
                              tf.shape(self.logits)[1])

        # In case the labeler is stacked on a decoder which emits also an end
        # symbol (or for some reason emits more symbol than we have in the
        # ground truth labels), we trim the sequences to the length of a
        # shorter one.

        # pylint: disable=unsubscriptable-object
        return tf.contrib.seq2seq.sequence_loss(
            logits=self.logits[:, :min_time],
            targets=self.train_targets[:, :min_time],
            weights=self.input_sequence.temporal_mask[:, :min_time])
        # pylint: enable=unsubscriptable-object

    @property
    def train_loss(self) -> tf.Tensor:
        return self.cost

    @property
    def runtime_loss(self) -> tf.Tensor:
        return self.cost

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        fd = ModelPart.feed_dict(self, dataset, train)

        sentences = dataset.maybe_get_series(self.data_id)
        if sentences is not None:
            vectors, paddings = self.vocabulary.sentences_to_tensor(
                list(sentences), pad_to_max_len=False, train_mode=train)

            fd[self.train_targets] = vectors.T
            fd[self.train_weights] = paddings.T

        return fd
