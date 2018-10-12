import os


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def freeze_network(network):
    for p in network.parameters():
        p.requires_grad = False


def unfreeze_network(network):
    for p in network.parameters():
        p.requires_grad = True
