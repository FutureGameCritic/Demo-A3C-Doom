import mxnet as mx
import mxnet.ndarray as F
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn

net = nn.HybridSequential(prefix='MLP_')
with net.name_scope():
    net.add(
        nn.Flatten(),
        nn.Dense(128, activation='relu'),
        nn.Dense(64, activation='relu'),
        nn.Dense(10, activation=None)  # loss function includes softmax already, see below
    )

ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu(0)
net.initialize(mx.init.Xavier(), ctx=ctx)

trainer = gluon.Trainer(
    params=net.collect_params(),
    optimizer='sgd',
    optimizer_params={'learning_rate': 0.04},
)

class Policy(gluon.Block):
    def __init__(self, **kwargs):
        super(Policy, self).__init__(**kwargs)
        with self.name_scope():
            self.dense = nn.Dense(16, in_units=kwargs.input_size, activation='relu')
            self.action_pred = nn.Dense(kwargs.action_size, in_units=16)
            self.value_pred = nn.Dense(kwargs.value_size, in_units=16)

    def forward(self, x):
        x = self.dense(x)
        probs = self.action_pred(x)
        values = self.value_pred(x)
        return F.softmax(probs), values

class a3c_global(object):
    def __init__(self, hparams, env):
        
        # env
        self.net = self.get_model(
                        input_size=16,
                        action_size=2,
                        value_size=1
                    )
        self.net.initialize(mx.init.Uniform(0.02))
        # trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 3e-2})
        # loss = gluon.loss.L1Loss()

    def get_model(self, input_size, action_size, value_size):
        return Policy(input_size=input_size, action_size=action_size, value_size=value_size)

class a3c_worker(a3c_global):
    def __init__(self, hparams):
        a3c_global.__init__(self, hparams)
        self.model = None