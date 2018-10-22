import os

import torch

from torchvision.utils import save_image


def generate_images(code_generator, decoder, save_path, epoch, device, images_number=10):
    torch.random.manual_seed(1986)
    z = torch.randn(images_number, code_generator.noise_size, 1, 1, requires_grad=True, device=device)

    samples = decoder(code_generator(z))

    filename = os.path.join(save_path, 'samples_{}.jpg'.format(epoch))
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    save_image(tensor=samples, filename=filename, nrow=4, padding=2,
               normalize=True, range=None, scale_each=True, pad_value=0)
