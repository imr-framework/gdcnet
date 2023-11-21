from tensorflow import keras as K
import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.initializers as KI

from gdcnet.model.gdcnet_arch import Unet, SpatialTransformer

class gdcnet_vm(K.Model):
    def __init__(self,
                 inshape,
                 arch='unet+stu',
                 name='gdcnet_vm'):
        '''
        Parameters
        ----------
        inshape : tuple
            Input shape of the model (64, 64) or (64,64,32).
        arch : str, optional
            Architecture of the model. The default is 'unet+stu' for the semi-supervised and self-supervised models. 'unet' for the supervised model.
        name : str, optional
            Name of the model. The default is 'gdcnet_vm'.
        '''

        # Check dimensions
        ndims = len(inshape)
        assert ndims in [2, 3], 'ndims should be 2, or 3. found: %d' % ndims

        source = tf.keras.Input(shape=(*inshape, 1), name='%s_source_input' % name)
        target = tf.keras.Input(shape=(*inshape, 1), name='%s_target_input' % name)
        input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])

        # Build the U-Net
        vdmnet = Unet(input_model=input_model,
                      nb_filters=[[16, 32, 32, 32],[32, 32, 32, 32, 32, 16, 16]],
                      name='%s_vdmnet' % name)

        # Get the VDM along the PE direction
        ndim_flow = 1
        Conv = getattr(KL, 'Conv%dD' % ndim_flow)
        VDM = Conv(ndim_flow, kernel_size=3, padding='same',
                    kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-5),
                    name='%s_vdm' % name)(vdmnet.output)

        if arch == 'unet+stu':
            # warp image with flow field (Spatial Transfor Unit)
            if ndims == 2:
                y_source = SpatialTransformer(interp_method='linear',
                                                     indexing='ij',
                                                     fill_value=None,
                                                     name='%s_transformer' % name)([source, tf.concat([VDM, tf.zeros_like(VDM)], axis=-1)])
            elif ndims==3:
                    y_source = SpatialTransformer(interp_method='linear',
                                                         indexing='ij',
                                                         fill_value=None,
                                                         name='%s_transformer' % name)([source, tf.concat([VDM, tf.zeros_like(VDM), tf.zeros_like(VDM)], axis=-1)])

            # initilize the keras model
            outputs = [y_source]
        elif arch == 'unet':
            outputs = []
        else:
            # Error
            raise ValueError(f'Unknown option "{arch}" for train_type.')

        outputs.append(VDM)
        super().__init__(inputs=input_model.inputs, outputs=outputs, name=name)