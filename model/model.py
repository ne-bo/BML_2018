from base import BaseModel
from model.networks.networks import Encoder, Decoder, CodeGenerator, ImageDiscriminator, CodeDiscriminator
from utils import freeze_network, unfreeze_network


class MnistModel(BaseModel):
    def __init__(self, input_image_channels=1, out_image_channels=1, code_size=8, noise_size=8, input_image_size=28):
        super(MnistModel, self).__init__()

        self.code_size = code_size
        self.noise_size = noise_size
        self.input_image_size = input_image_size

        self.encoder = Encoder(code_size=code_size, input_image_channels=input_image_channels,
                               input_image_size=input_image_size)
        self.decoder = Decoder(code_size=code_size, out_image_channels=out_image_channels)
        self.code_generator = CodeGenerator(code_size=code_size, noise_size=noise_size)
        self.d_i = ImageDiscriminator(input_image_channels=input_image_channels,
                                      input_image_size=input_image_size)
        self.d_c = CodeDiscriminator(code_size=code_size)

    # here the input x is an image, and the input z is a gaussian noise
    def forward(self, x, z, phase=None):
        assert phase in ['AAE', 'PriorImprovement']

        assert z.shape[1] == self.noise_size

        z_c = self.code_generator(z)
        latent_code = self.encoder(x)
        x_rec = self.decoder(self.encoder(x))
        assert x.shape == x_rec.shape

        # In one phase, termed the prior improvement phase,
        # we update the code generator with the loss function in Eq. (4), by fixing the encoder
        if phase == 'PriorImprovement':
            unfreeze_network(self.code_generator)
            freeze_network(self.encoder)
            # pass the noise
            dec_z_c = self.decoder(z_c)
            d_i_dec_z_c, _, _ = self.d_i(dec_z_c)
            # pass the image
            d_i_x, _, _ = self.d_i(self.decoder(latent_code))
            d_i_x_rec, _, _ = self.d_i(x_rec)
            # return vectors we need to use inside the loss
            return d_i_x, d_i_dec_z_c, d_i_x_rec

        # In the other phase, termed the AAE phase,
        # we fix the code generator and update the autoencoder following the training procedure of AAE.
        if phase == 'AAE':
            unfreeze_network(self.encoder)
            freeze_network(self.code_generator)
            # pass the noise
            d_c_z_c = self.d_c(z_c)
            # pass the image
            d_c_enc_x = self.d_c(latent_code)
            # return vectors we need to use inside the loss
            return d_c_enc_x, d_c_z_c, x_rec


class CifarModel(BaseModel):
    def __init__(self, input_image_channels=3, out_image_channels=3, code_size=8, noise_size=8, input_image_size=32):
        super(CifarModel, self).__init__()

        self.code_size = code_size
        self.noise_size = noise_size
        self.input_image_size = input_image_size

        self.encoder = Encoder(code_size=code_size, input_image_channels=input_image_channels,
                               input_image_size=input_image_size)
        self.decoder = Decoder(code_size=code_size, out_image_channels=out_image_channels,
                               output_image_size=input_image_size)
        self.code_generator = CodeGenerator(code_size=code_size, noise_size=noise_size)
        self.d_i = ImageDiscriminator(input_image_channels=input_image_channels,
                                      input_image_size=input_image_size)
        self.d_c = CodeDiscriminator(code_size=code_size)

    # here the input x is an image, and the input z is a gaussian noise
    def forward(self, x, z, phase=None):
        assert phase in ['AAE', 'PriorImprovement']

        assert z.shape[1] == self.noise_size

        z_c = self.code_generator(z)
        latent_code = self.encoder(x)
        x_rec = self.decoder(self.encoder(x))
        assert x.shape == x_rec.shape, 'x {} x_rec {}'.format(x.shape, x_rec.shape)

        # In one phase, termed the prior improvement phase,
        # we update the code generator with the loss function in Eq. (4), by fixing the encoder
        if phase == 'PriorImprovement':
            unfreeze_network(self.code_generator)
            freeze_network(self.encoder)
            # pass the noise
            dec_z_c = self.decoder(z_c)
            d_i_dec_z_c, _, _ = self.d_i(dec_z_c)
            # pass the image
            d_i_x, _, _ = self.d_i(self.decoder(latent_code))
            d_i_x_rec, _, _ = self.d_i(x_rec)
            # return vectors we need to use inside the loss
            return d_i_x, d_i_dec_z_c, d_i_x_rec

        # In the other phase, termed the AAE phase,
        # we fix the code generator and update the autoencoder following the training procedure of AAE.
        if phase == 'AAE':
            unfreeze_network(self.encoder)
            freeze_network(self.code_generator)
            # pass the noise
            d_c_z_c = self.d_c(z_c)
            # pass the image
            d_c_enc_x = self.d_c(latent_code)
            # return vectors we need to use inside the loss
            return d_c_enc_x, d_c_z_c, x_rec
