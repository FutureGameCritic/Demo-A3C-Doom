import mxnet as mx
import mxnet.ndarray as F
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
from mxnet.gluon.loss import Loss
class ConvBlock(gluon.Block):
    def __init__(self, **kwargs):
        super(Policy, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = nn.Conv2D(16, 8, 4, activation='relu')
            self.conv2 = nn.Conv2D(32, 4, 2, activation='relu')
            self.flatten = nn.Flatten()
            self.fc = nn.Dense(256, activation='relu')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        return self.fc(x)

class CrossEntropy(Loss):
    def __init__(self, _weight = None, _batch_axis= 0, **kwards):
        Loss.__init__(self,weight=_weight, batch_axis = _batch_axis, **kwards)

    def hybrid_forward(self, action, prob, advantage):
        action_prob = F.sum(action * prob, axis=1)
        cross_entropy = F.log(action_prob) * advantage
        loss = -F.sum(cross_entropy)

        return loss

class A2C(object):
    def __init__(self, hparams, action_size, value_size):
        self.hparams = hparams

        self.net_actor, self.net_critic = self.build_model()
        self.trainer_actor, self.loss_actor, self.trainer_critic, self.loss_critic = self.get_optimizer()

    def build_model(self):
        actor = ConvBlock()
        actor.add(nn.Dense(self.hparams.action_size, activation='softmax'))
        actor.initialize(mx.init.Uniform(0.02))

        critic = ConvBlock()
        critic.add(nn.Dense(self.hparams.value_size, activation='linear'))
        critic.initialize(mx.init.Uniform(0.02))

        return actor, critic

    def get_optimizer(self):
        trainer_actor = gluon.Trainer(self.net_actor.collect_params(), 'adam', {'learning_rate': self.hparams.actor_learning_rate})
        loss_actor = CrossEntropy()

        trainer_critic = gluon.Trainer(self.net_critic.collect_params(), 'adam', {'learning_rate': self.hparams.critic_learning_rate})
        loss_critic = gluon.loss.L1Loss()
        
        return trainer_actor, loss_actor, trainer_critic, loss_critic

    def get_state_from_game(state):
        out_state = state.screen_buffer
        return out_state
        
    def get_action(self, state):
        policy = self.net_actor(state)
        return np.random.choice(self.action_size, 1, p=policy)[0]

    def train_step(self, state, action, reward, next_state, done):
        value = self.net_critic(state)
        next_value = self.net_critic(next_state)
        
        # one-hot encoding
        act = np.zeros([1, self.action_size])
        act[0][action] = 1

        if done:
            advantage = reward - value
            target = [reward]
        else:
            advantage = (reward + self.hparams.discount_factor * next_value) - value
            target = reawrd + self.hparams.discount_factor * next_value
        
        with autograd.record():
            # update actor
            La = self.loss_actor(act, self.net_actor(state), advantage)
            La.backward()
            # update critic
            Lc = self.loss_critic(value, target)
            Lc.backward()

        # step to trainer
        self.trainer_actor.step(1)
        self.trainer_critic.step(1)