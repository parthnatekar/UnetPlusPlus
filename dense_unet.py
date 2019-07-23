from segmentation_models import Unet, Nestnet, Xnet, PSPNet
from losses import *
from keras.optimizers import SGD, Adam


class Dense_Unet():

    def __init__(self, img_shape):
        self.img_shape = img_shape

    def compile_dense(self):

        self.model = Unet(backbone_name='inceptionv3',
                 input_shape=self.img_shape,
                 input_tensor=None,
                 encoder_weights=None,
                 freeze_encoder=False,
                 skip_connections='default',
                 decoder_block_type='upsampling',
                 decoder_filters=(256,128,64,32,16),
                 decoder_use_batchnorm=True,
                 n_upsample_blocks=5,
                 upsample_rates=(2,2,2,2,2),
                 classes=4,
                 activation='softmax')

        sgd = SGD(lr=0.1, momentum=0.9, decay=5e-6, nesterov=False)
        adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        self.model.compile(adam, gen_dice_loss, [dice_whole_metric,dice_core_metric,dice_en_metric])

        return(self.model)

