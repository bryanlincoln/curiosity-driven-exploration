from chainer import Link, Chain, ChainList
import chainer.links as L
import chainer.functions as F
import chainerrl


class RealPPOModel(Chain):
    def __init__(self, n_actions):
        super(RealPPOModel, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 32, 8, stride=4)
            self.conv2 = L.Convolution2D(None, 64, 4, stride=2)
            #self.conv3 = L.Convolution2D(None, 64, 3, stride=1)
            self.l1 = L.Linear(None, 512)
            self.l2_pi = L.Linear(None, 256)
            self.l2_val = L.Linear(None, 256)
            self.pi = L.Linear(None, n_actions)
            self.val = L.Linear(None, 1)
            self.gaussianPolicy = chainerrl.policies.GaussianHeadWithStateIndependentCovariance(
                action_size=n_actions,
                var_type='diagonal',
                var_func=lambda x: F.exp(2 * x),  # Parameterize log std
                var_param_init=0,  # log std = 0 => std = 1
            )

    def forward(self, x):
        # shared layers
        im = x['retina']
        im = F.relu(self.conv1(im))
        im = F.relu(self.conv2(im))
        #im = F.relu(self.conv3(im))
        im = self.l1(im)
        imx = F.concat([im, x['joint_positions'], x['touch_sensors']])

        # pi layers
        l2_pi = F.relu(self.l2_pi(imx))
        pi = self.pi(l2_pi)
        pi = self.gaussianPolicy(pi)

        # value layers
        value = F.relu(self.l2_val(imx))
        value = self.val(value)
        return pi, value
