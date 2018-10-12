from torch import optim

from utils import get_instance


def get_trainable_params(network):
    return filter(lambda p: p.requires_grad, network.parameters())


class OurOptimizersAndSchedulers():
    def __init__(self, model, config):
        super(OurOptimizersAndSchedulers, self).__init__()

        self.encoder_optimizer = get_instance(optim, 'optimizer', config,
                                              get_trainable_params(model.encoder))
        self.decoder_optimizer = get_instance(optim, 'optimizer', config,
                                              get_trainable_params(model.decoder))
        self.code_generator_optimizer = get_instance(optim, 'optimizer', config,
                                                     get_trainable_params(model.code_generator))
        self.d_i_optimizer = get_instance(optim, 'optimizer', config,
                                          get_trainable_params(model.d_i))
        self.d_c_optimizer = get_instance(optim, 'optimizer', config,
                                          get_trainable_params(model.d_c))

        self.encoder_scheduler = get_instance(optim.lr_scheduler, 'lr_scheduler', config,
                                              self.encoder_optimizer)
        self.decoder_scheduler = get_instance(optim.lr_scheduler, 'lr_scheduler', config,
                                              self.decoder_optimizer)
        self.code_generator_scheduler = get_instance(optim.lr_scheduler, 'lr_scheduler', config,
                                                     self.code_generator_optimizer)
        self.d_i_scheduler = get_instance(optim.lr_scheduler, 'lr_scheduler', config,
                                          self.d_i_optimizer)
        self.d_c_scheduler = get_instance(optim.lr_scheduler, 'lr_scheduler', config,
                                          self.d_c_optimizer)
