import tensorflow as tf
import argparse

class loader():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='parameters')
        self.hparams = dict()

    def add_argument(self, name, type, default, help):
        self.parser.add_argument('--'+name, type=type, help=help)
        self.hparams[name] = default
    
    def parsing(self):
        args = self.parser.parse_args()
        cmd = ",".join(['{}={}'.format(arg, getattr(args, arg)) for arg in vars(args)])

        hparams = tf.contrib.training.HParams(**self.hparams)
        hparams.parse(cmd)

        return hparams
        
