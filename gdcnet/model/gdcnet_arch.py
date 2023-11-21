import warnings
import keras.backend as K
import tensorflow as tf
import tensorflow.keras.layers as KL

from gdcnet.model.utils import _conv_block, _upsample_block, is_affine_shape, affine_to_dense_shift,transform


class Unet(tf.keras.Model):
    def __init__(self,
                 input_model,
                 nb_filters,
                 kernel_initializer='he_normal',
                 name='vdmnet'):
        '''
        Parameters
        ----------
        input_model : tf.keras.Model
            Model to be used as input to the U-Net.
        nb_filters: list
            Number of filters for each layer of the U-Net structured as [[encoder filters], [decoder filters]].
        kernel_initializer: str, optional
            Kernel initializer to be used for the convolutional layers. The default is 'he_normal'.
        name : str, optional
            Name of the model. The default is 'vdmnet'.
        '''
        if len(input_model.outputs) == 1:
            unet_input = input_model.outputs[0]
        else:
            unet_input = KL.concatenate(input_model.outputs, name='%s_input_concat' % name)
        model_inputs = input_model.inputs

        ndims = len(unet_input.get_shape()) - 2
        assert ndims in (2, 3), 'ndims should be 2, or 3. found: %d' % ndims
        MaxPooling = getattr(KL, 'MaxPooling%dD' % ndims)

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_filters
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        nb_levels = int(nb_dec_convs) + 1
        max_pool = [2] * nb_levels

        # ENCODER
        enc_layers = []
        last = unet_input
        for level in range(nb_levels - 1):

            nf = enc_nf[level]
            layer_name = '%s_enc_conv_%d' % (name, level)
            last = _conv_block(last, nf, name=layer_name, kernel_initializer=kernel_initializer)
            enc_layers.append(last)

            # temporarily use maxpool since downsampling doesn't exist in keras
            last = MaxPooling(max_pool[level], name='%s_enc_pooling_%d' % (name, level))(last)

        # activate = lambda lvl, c: True

        # DECODER
        for level in range(nb_levels - 1):
            real_level = nb_levels - level - 2
            nf = dec_nf[level]
            layer_name = '%s_dec_conv_%d' % (name, real_level)
            last = _conv_block(last, nf, name=layer_name, kernel_initializer=kernel_initializer)

            # upsample
            if level < (nb_levels - 1):
                layer_name = '%s_dec_upsample_%d' % (name, real_level)
                last = _upsample_block(last, enc_layers.pop(), factor=max_pool[real_level],
                                       name=layer_name)


        # FINAL CONVOLUTIONS
        for i, nf in enumerate(final_convs):
            layer_name = '%s_final_conv_%d' % (name, i)
            last = _conv_block(last, nf, name=layer_name, kernel_initializer=kernel_initializer)

        super().__init__(inputs=model_inputs, outputs=last, name=name)

from keras.layers import Layer
class SpatialTransformer(Layer):
    """
    ND spatial transformer layer
    Applies affine and dense transforms to images. A dense transform gives
    displacements (not absolute locations) at each voxel.
    If you find this layer useful, please cite:
      Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
      Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
      MICCAI 2018.
    Originally, this code was based on voxelmorph code, which
    was in turn transformed to be dense with the help of (affine) STN code
    via https://github.com/kevinzakka/spatial-transformer-network.
    Since then, we've re-written the code to be generalized to any
    dimensions, and along the way wrote grid and interpolation functions.
    """

    def __init__(self,
                 interp_method='linear',
                 indexing='ij',
                 single_transform=False,
                 fill_value=None,
                 shift_center=True,
                 shape=None,
                 **kwargs):
        """
        Parameters:
            interp_method: Interpolation method. Must be 'linear' or 'nearest'.
            indexing: Must be 'ij' (matrix) or 'xy' (cartesian). 'xy' indexing will
                have the first two entries of the flow (along last axis) flipped
                compared to 'ij' indexing.
            single_transform: Use single transform for the entire image batch.
            fill_value: Value to use for points sampled outside the domain.
                If None, the nearest neighbors will be used.
            shift_center: Shift grid to image center when converting affine
                transforms to dense transforms.
            shape: ND output shape used when converting affine transforms to dense
                transforms. Includes only the N spatial dimensions. If None, the
                shape of the input image will be used.
        """
        self.interp_method = interp_method
        assert indexing in ['ij', 'xy'], "indexing has to be 'ij' (matrix) or 'xy' (cartesian)"
        self.indexing = indexing
        self.single_transform = single_transform
        self.fill_value = fill_value
        self.shift_center = shift_center
        self.shape = shape
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'interp_method': self.interp_method,
            'indexing': self.indexing,
            'single_transform': self.single_transform,
            'fill_value': self.fill_value,
            'shift_center': self.shift_center,
            'shape': self.shape,
        })
        return config

    def build(self, input_shape):

        # sanity check on input list
        if len(input_shape) > 2:
            raise ValueError('Spatial Transformer must be called on a list of length 2: '
                             'first argument is the image, second is the transform.')

        # set up number of dimensions
        self.ndims = len(input_shape[0]) - 2
        self.imshape = input_shape[0][1:]
        self.trfshape = input_shape[1][1:]
        self.is_affine = is_affine_shape(input_shape[1][1:])

        # make sure inputs are reasonable shapes
        if self.is_affine:
            expected = (self.ndims, self.ndims + 1)
            actual = tuple(self.trfshape[-2:])
            if expected != actual:
                raise ValueError(f'Expected {expected} affine matrix, got {actual}.')
        else:
            image_shape = tuple(self.imshape[:-1])
            dense_shape = tuple(self.trfshape[:-1])
            if image_shape != dense_shape:
                warnings.warn(f'Dense transform shape {dense_shape} does not match '
                              f'image shape {image_shape}.')

        # confirm built
        self.built = True

    def call(self, inputs):
        """
        Parameters
            inputs: List of [img, trf], where img is the ND moving image and trf
            is either a dense warp of shape [B, D1, ..., DN, N] or an affine matrix
            of shape [B, N, N+1].
        """

        # necessary for multi-gpu models
        vol = K.reshape(inputs[0], (-1, *self.imshape))
        trf = K.reshape(inputs[1], (-1, *self.trfshape))

        # convert affine matrix to warp field
        if self.is_affine:
            shape = vol.shape[1:-1] if self.shape is None else self.shape
            fun = lambda x: affine_to_dense_shift(x, shape,
                                                        shift_center=self.shift_center,
                                                        indexing=self.indexing)
            trf = tf.map_fn(fun, trf)

        # prepare location shift
        if self.indexing == 'xy':  # shift the first two dimensions
            trf_split = tf.split(trf, trf.shape[-1], axis=-1)
            trf_lst = [trf_split[1], trf_split[0], *trf_split[2:]]
            trf = tf.concat(trf_lst, -1)

        # map transform across batch
        if self.single_transform:
            return tf.map_fn(lambda x: self._single_transform([x, trf[0, :]]), vol)
        else:
            return tf.map_fn(self._single_transform, [vol, trf], fn_output_signature=vol.dtype)

    def _single_transform(self, inputs):
        return transform(inputs[0], inputs[1], interp_method=self.interp_method,
                               fill_value=self.fill_value)