"""An implementation of Conv-Highway Model."""
import typing

import keras
from keras import backend as K

from matchzoo import engine
from matchzoo import preprocessors

class ConvHighway(engine.BaseModel):
    """
    ConvHighway Model.

    Examples:
        >>> model = ConvHighway()
        >>> model.params['encode_filters'] = 128
        >>> model.params['encode_kernel_sizes'] = [3, 4, 5]
        >>> model.params['decode_filters'] = 128
        >>> model.params['decode_kernel_size'] = 3
        >>> model.params['conv_activation_func'] = 'relu'
        >>> model.params['pool_size'] = 2
        >>> model.params['mlp_num_layers'] = 1
        >>> model.params['mlp_num_units'] = 200
        >>> model.params['mlp_num_fan_out'] = 100
        >>> model.params['mlp_activation_func'] = 'relu'
        >>> model.params['dropout_rate'] = 0.5
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls) -> engine.ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(
            with_embedding=True,
            with_multi_layer_perceptron=True
        )
        params['optimizer'] = 'adam'
        params.add(engine.Param(name='encode_filters', value=128,
                                desc="The filter size of each convolution "
                                     "blocks for encode."))
        params.add(engine.Param(name='encode_kernel_sizes', value=[3],
                                desc="The kernel size of each convolution "
                                     "blocks for the encode."))
        params.add(engine.Param(name='decode_filters', value=128,
                                desc="The filter size of each convolution "
                                     "blocks for decode."))
        params.add(engine.Param(name='decode_kernel_size', value=3,
                                desc="The kernel size of each convolution "
                                     "blocks for the decode."))
        params.add(engine.Param(name='conv_activation_func', value='relu',
                                desc="The activation function in the "
                                     "convolution layer."))
        params.add(engine.Param(name='pool_size', value=4,
                                desc="pool size of max pooling"
                                     "pool size of max pooling"))
        params.add(engine.Param(
            name='padding',
            value='same',
            hyper_space=engine.hyper_spaces.choice(
                ['same', 'valid', 'causal']),
            desc="The padding mode in the convolution layer. It should be one"
                 "of `same`, `valid`, and `causal`."
        ))
        params.add(engine.Param(
            'dropout_rate', 0.0,
            hyper_space=engine.hyper_spaces.quniform(
                low=0.0, high=0.8, q=0.01),
            desc="The dropout rate."
        ))
        return params

    def build(self):
        """
        Build model structure.

        ConvHighway use Siamese arthitecture.
        """
        input_left, input_right = self._make_inputs()

        mask_left = keras.layers.Lambda(lambda x: K.cast(x, K.tf.bool))(input_left)
        mask_right = keras.layers.Lambda(lambda x: K.cast(x, K.tf.bool))(input_right)

        embedding = self._make_embedding_layer()
        embed_left = embedding(input_left)
        embed_right = embedding(input_right)

        # encode
        encode_left = self._conv_highway_block(
            embed_left,
            self._params['encode_filters'],
            self._params['encode_kernel_sizes'],
            self._params['padding'],
            self._params['conv_activation_func']
        )
        encode_right = self._conv_highway_block(
            embed_right,
            self._params['encode_filters'],
            self._params['encode_kernel_sizes'],
            self._params['padding'],
            self._params['conv_activation_func']
        )

        attention_layer = keras.layers.Lambda(ConvHighway.bi_attention)
        attn = attention_layer([encode_left, encode_right, mask_left, mask_right])
        # decode
        decode = self._conv_pool_block(
            attn,
            self._params['decode_filters'],
            self._params['decode_kernel_size'],
            self._params['padding'],
            self._params['conv_activation_func'],
            self._params['pool_size']
        )
        output = keras.layers.Flatten()(decode)
        dropout = keras.layers.Dropout(
            rate=self._params['dropout_rate'])(output)
        mlp = self._make_multi_layer_perceptron_layer()(dropout)

        inputs = [input_left, input_right]
        x_out = self._make_output_layer()(mlp)
        self._backend = keras.Model(inputs=inputs, outputs=x_out)

    def _conv_pool_block(
        self,
        input_: typing.Any,
        filters: int,
        kernel_size: int,
        padding: str,
        conv_activation_func: str,
        pool_size: int
    ) -> typing.Any:
        output = keras.layers.Conv1D(
            filters,
            kernel_size,
            padding=padding,
            activation=conv_activation_func
        )(input_)
        output = keras.layers.MaxPooling1D(pool_size=pool_size)(output)
        return output
        
    def _conv_highway_block(
        self,
        input_: typing.Any,
        filters: int,
        kernel_sizes: list,
        padding: str,
        conv_activation_func: str,
    ) -> typing.Any:
        inputs = keras.layers.Dense(filters)(input_)
        shortconn = inputs
        for kidx, kernel_size in enumerate(kernel_sizes, 1):
            if kidx % 2 == 0:
                inputs = keras.layers.Add()([inputs, shortconn])
                shortconn = inputs
            else:
                H = keras.layers.Conv1D(
                    filters,
                    kernel_size,
                    padding=padding,
                    activation=conv_activation_func
                )(inputs)
                T = keras.layers.Conv1D(
                    filters,
                    kernel_size,
                    padding=padding,
                    activation='sigmoid'
                )(inputs)
                inputs = keras.layers.Add()([
                    keras.layers.Multiply()([H, T]),
                    keras.layers.Multiply()([inputs, keras.layers.Lambda(lambda x: 1.0-x)(T)])
                ])
        return inputs

    @staticmethod
    def bi_attention(
        tensors: typing.Any
    ) -> typing.Any:
        p_enc, q_enc, p_mask, q_mask = tensors
        score = ConvHighway.bilinear(p_enc, q_enc)
        q_mask_ex = K.expand_dims(q_mask, 1) # batch x 1 x q_len
        p_mask_ex = K.expand_dims(p_mask, 1) # batch x 1 x p_len
        
        score_ = K.softmax(
            K.expand_dims(
                K.tf.reduce_max(
                    ConvHighway.mask_logits(score, p_mask_ex), axis=1), axis=1), -1) # batch x 1 x p_len
        #score_x = K.tf.tile(score_, [1, K.tf.shape(p_enc)[1], 1])
        score_t = K.softmax(
            ConvHighway.mask_logits(K.tf.transpose(score, (0, 2, 1)), q_mask_ex)
        ) # batch x p_len x q_len
        p2q = K.batch_dot(score_, p_enc) # batch x 1 x embedding_size
        q2p = K.batch_dot(score_t, q_enc) # batch x p_len x embedding_size

        concat = K.tf.concat([p_enc, q2p, p_enc*p2q, p_enc*q2p], axis=-1)
        return concat

    @staticmethod
    def mask_logits(
        inputs: typing.Any,
        mask: typing.Any
    ) -> typing.Any:
        mask = K.cast(mask, K.tf.float32)
        return inputs * mask + 1e-12 * (1. - mask)

    @staticmethod
    def bilinear(
        p_enc: typing.Any,
        q_enc: typing.Any
    ) -> typing.Any:
        """
        Args:
            p_enc: (batch_size, p_len, embed_size)
            q_enc: (batch_size, q_len, embed_size)
        
        Ouput: 
            (batch_size, p_len, q_len)
        """
        p = K.tf.transpose(p_enc, (0, 2, 1))
        with K.tf.variable_scope("attn_weight", reuse=K.tf.AUTO_REUSE):
            hidden_dim = q_enc.get_shape()[-1]
            attn_W = K.tf.get_variable("AttnW",
                                shape=[hidden_dim, hidden_dim],
                                dtype=K.tf.float32)
            w_q = K.dot(q_enc, attn_W)
        out = K.batch_dot(w_q, p)  # batch x q_len x p_len
        return out
