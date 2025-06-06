"""
Dense Convolutional Encoder-Decoder Networks

Reference:
    https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

"""

import torch
import torch.nn as nn
import torchvision

class _DenseLayer(nn.Sequential):
    """One dense layer within dense block, with bottleneck design.

    Args:
        in_features (int):
        growth_rate (int): # out feature maps of every dense layer
        drop_rate (float): 
        bn_size (int): Specifies maximum # features is `bn_size` * 
            `growth_rate`
        bottleneck (bool, False): If True, enable bottleneck design
    """
    def __init__(self, in_features, growth_rate, drop_rate=0., bn_size=8,
                 bottleneck=False):
        super(_DenseLayer, self).__init__()
        if bottleneck and in_features > bn_size * growth_rate:
            self.add_module('norm1', nn.BatchNorm2d(in_features))
            self.add_module('relu1', nn.ReLU(inplace=True))
            self.add_module('conv1', nn.Conv2d(in_features, bn_size *
                            growth_rate, kernel_size=1, stride=1, bias=False))
            self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
            self.add_module('relu2', nn.ReLU(inplace=True))
            self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                            kernel_size=3, stride=1, padding=1, bias=False))
        else:
            self.add_module('norm1', nn.BatchNorm2d(in_features))
            self.add_module('relu1', nn.ReLU(inplace=True))
            self.add_module('conv1', nn.Conv2d(in_features, growth_rate,
                            kernel_size=3, stride=1, padding=1, bias=False))
        if drop_rate > 0:
            self.add_module('dropout', nn.Dropout2d(p=drop_rate))
        
    def forward(self, x):
        y = super(_DenseLayer, self).forward(x)
        return torch.cat([x, y], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_features, growth_rate, drop_rate,
                 bn_size=4, bottleneck=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(in_features + i * growth_rate, growth_rate,
                                drop_rate=drop_rate, bn_size=bn_size,
                                bottleneck=bottleneck)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, in_features, out_features, down, bottleneck=True, 
                 drop_rate=0):
        """Transition layer, either downsampling or upsampling, both reduce
        number of feature maps, i.e. `out_features` should be less than 
        `in_features`.

        Args:
            in_features (int):
            out_features (int):
            down (bool): If True, downsampling, else upsampling
            bottleneck (bool, True): If True, enable bottleneck design
            drop_rate (float, 0.):
        """
        super(_Transition, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(in_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        if down:
            # half feature resolution, reduce # feature maps
            if bottleneck:
                # bottleneck impl, save memory, add nonlinearity
                self.add_module('conv1', nn.Conv2d(in_features, out_features,
                    kernel_size=1, stride=1, padding=0, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout2d(p=drop_rate))
                self.add_module('norm2', nn.BatchNorm2d(out_features))
                self.add_module('relu2', nn.ReLU(inplace=True))
                # self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
                # not using pooling, fully convolutional...
                self.add_module('conv2', nn.Conv2d(out_features, out_features,
                    kernel_size=3, stride=2, padding=1, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout2', nn.Dropout2d(p=drop_rate))
            else:
                self.add_module('conv1', nn.Conv2d(in_features, out_features,
                    kernel_size=3, stride=2, padding=1, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout2d(p=drop_rate))
        else:  # If up
            # transition up, increase feature resolution, half # feature maps
            if bottleneck:
                # bottleneck impl, save memory, add nonlinearity
                self.add_module('conv1', nn.Conv2d(in_features, out_features,
                    kernel_size=1, stride=1, padding=0, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout2d(p=drop_rate))

                self.add_module('norm2', nn.BatchNorm2d(out_features))
                self.add_module('relu2', nn.ReLU(inplace=True))
                # output_padding=0, or 1 depends on the image size
                # if image size is of the power of 2, then 1 is good
                # Originally the paddings are both 1's
                self.add_module('convT2', nn.ConvTranspose2d(
                    out_features, out_features, kernel_size=3, stride=2,
                    padding=1, output_padding=1, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout2', nn.Dropout2d(p=drop_rate))
            else:
                self.add_module('convT1', nn.ConvTranspose2d(
                    out_features, out_features, kernel_size=3, stride=2,
                    padding=1, output_padding=1, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout2d(p=drop_rate))


def last_decoding(in_features, out_channels, kernel_size, stride, padding, 
                  output_padding=0, bias=False, drop_rate=0.):
    """Last transition up layer, which outputs directly the predictions.
    """
    last_up = nn.Sequential()
    last_up.add_module('norm1', nn.BatchNorm2d(in_features))
    last_up.add_module('relu1', nn.ReLU(True))
    last_up.add_module('conv1', nn.Conv2d(in_features, in_features // 2, 
                    kernel_size=1, stride=1, padding=0, bias=False))
    if drop_rate > 0.:
        last_up.add_module('dropout1', nn.Dropout2d(p=drop_rate))
    last_up.add_module('norm2', nn.BatchNorm2d(in_features // 2))
    last_up.add_module('relu2', nn.ReLU(True))
    last_up.add_module('convT2', nn.ConvTranspose2d(in_features // 2, 
                       out_channels, kernel_size=kernel_size, stride=stride, 
                       padding=padding, output_padding=output_padding, bias=bias))
    return last_up


def activation(name, *args):
    if name in ['tanh', 'Tanh']:
        return nn.Tanh()
    elif name in ['relu', 'ReLU']:
        return nn.ReLU(inplace=True)
    elif name in ['lrelu', 'LReLU']:
        return nn.LeakyReLU(inplace=True)
    elif name in ['sigmoid', 'Sigmoid']:
        return nn.Sigmoid()
    elif name in ['softplus', 'Softplus']:
        return nn.Softplus(beta=4)
    else:
        raise ValueError('Unknown activation function')


class DenseED(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, blocks=[1,1,1,1,1,1,1], growth_rate=16,
                 init_features=6, in_conv_ker_size=21, in_dilation=1, in_batch_norm=False, in_groups=1, in_bias=False, in_combine=False, in_combine_ker_size=3, out_conv_ker_size=4, bn_size=8, drop_rate=0, bottleneck=False,
                 out_activation=None):
        """Dense Convolutional Encoder-Decoder Networks.

        In the network presented in the paper, the last decoding layer 
        (transition up) directly outputs the predicted fields. 

        The network parameters should be modified for different image size,
        mostly the first conv and the last convT layers. (`output_padding` in
        ConvT can be modified as well)

        Args:
            in_channels (int): number of input channels (also include time if
                time enters in the input)
            out_channels (int): number of output channels
            blocks (list-like): A list (of odd size) of integers
            growth_rate (int): K
            init_features (int): the number of feature maps after the first
                conv layer
            in_ker_size (int): the size of kernel in initial convolution
            bn_size: bottleneck size for number of feature maps
            bottleneck (bool): use bottleneck for dense block or not
            drop_rate (float): dropout rate
            out_activation: Output activation function, choices=[None, 'tanh',
                'sigmoid', 'softplus']
        """
        super(DenseED, self).__init__()
        if len(blocks) > 1 and len(blocks) % 2 == 0:
            raise ValueError('length of blocks must be an odd number, but got {}'
                            .format(len(blocks)))
        enc_block_layers = blocks[: len(blocks) // 2]
        dec_block_layers = blocks[len(blocks) // 2:]

        self.features = nn.Sequential()

        # First convolution, half image size ================
        # For even image size: k7s2p3, k5s2p2
        # For odd image size (e.g. 65): k7s2p2, k5s2p1, k13s2p5, k11s2p4, k9s2p3

        if in_combine:
            self.features.add_module('In_combine', nn.Conv2d(in_channels, 1, 
                        kernel_size=in_combine_ker_size, stride=1, padding=int((in_combine_ker_size-1)/2), bias=True))

            if in_batch_norm:
                self.features.add_module('In_norm', nn.BatchNorm2d(1))

            self.features.add_module('In_conv', nn.Conv2d(1, init_features, 
                            kernel_size=in_conv_ker_size, stride=2, dilation=in_dilation, padding=int(in_dilation*(in_conv_ker_size-1)/2), groups=in_groups, bias=in_bias))     
        else:
            if in_batch_norm :
                self.features.add_module('In_norm', nn.BatchNorm2d(in_channels))

            self.features.add_module('In_conv', nn.Conv2d(in_channels, init_features, 
                            kernel_size=in_conv_ker_size, stride=2, dilation=in_dilation, padding=int(in_dilation*(in_conv_ker_size-1)/2), groups=in_groups, bias=in_bias))

        # Encoding / transition down ================
        # dense block --> encoding --> dense block --> encoding
        num_features = init_features
        for i, num_layers in enumerate(enc_block_layers):
            block = _DenseBlock(num_layers=num_layers,
                                in_features=num_features,
                                bn_size=bn_size, 
                                growth_rate=growth_rate,
                                drop_rate=drop_rate,
                                bottleneck=bottleneck)
            self.features.add_module('EncBlock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate

            trans_down = _Transition(in_features=num_features,
                                     out_features=num_features // 2,
                                     down=True, 
                                     drop_rate=drop_rate)
            self.features.add_module('TransDown%d' % (i + 1), trans_down)
            num_features = num_features // 2
        # Decoding / transition up ==============
        # dense block --> decoding --> dense block --> decoding --> dense block
        for i, num_layers in enumerate(dec_block_layers):
            block = _DenseBlock(num_layers=num_layers,
                                in_features=num_features,
                                bn_size=bn_size, 
                                growth_rate=growth_rate,
                                drop_rate=drop_rate, 
                                bottleneck=bottleneck)
            self.features.add_module('DecBlock%d' % (i + 1), block)
            num_features += num_layers * growth_rate
            # the last decoding layer has different convT parameters
            if i < len(dec_block_layers) - 1:
                trans_up = _Transition(in_features=num_features,
                                    out_features=num_features // 2,
                                    down=False, 
                                    drop_rate=drop_rate)
                self.features.add_module('TransUp%d' % (i + 1), trans_up)
                num_features = num_features // 2
        
        # The last decoding layer =======
        # Originally output padding=1, stride=2
        last_trans_up = last_decoding(num_features, out_channels, 
                            kernel_size=out_conv_ker_size, stride=2, padding=out_conv_ker_size//2, 
                            output_padding=1, bias=False, drop_rate=drop_rate)
        self.features.add_module('LastTransUp', last_trans_up)
        if out_activation is not None:
            self.features.add_module(out_activation, activation(out_activation))
        


    def forward(self, x):
        return self.features(x)

    def init_print(self):
        print('# params {}, # conv layers {}'.format(
            *self._num_parameters_convlayers()))

    def forward_test(self, x):
        print('input: {}'.format(x.data.size()))
        for name, module in self.features._modules.items():
            x = module(x)
            print('{}: {}'.format(name, x.data.size()))
        return x


    def _num_parameters_convlayers(self):
        n_params, n_conv_layers = 0, 0
        for name, param in self.named_parameters():
            if 'conv' in name:
                n_conv_layers += 1
            n_params += param.numel()
        return n_params, n_conv_layers

    def _count_parameters(self):
        n_params = 0
        for name, param in self.named_parameters():
            print(name)
            print(param.size())
            print(param.numel())
            n_params += param.numel()
            print('num of parameters so far: {}'.format(n_params))

    def reset_parameters(self, verbose=False):
        for module in self.modules():
            # pass self, otherwise infinite loop
            if isinstance(module, self.__class__):
                continue
            if 'reset_parameters' in dir(module):
                if callable(module.reset_parameters):
                    module.reset_parameters()
                    if verbose:
                        print("Reset parameters in {}".format(module))


def example_init():
    # initialize model
    device = "cuda:0"
    in_channels = 1
    out_channels = 1
    blocks = [1, 1, 1, 1, 1, 1, 1]
    growth_rate = 16
    init_features = 6
    in_conv_ker_size = 21
    out_conv_ker_size = 5
    in_dilation = 1
    in_batch_norm = True
    in_groups = 1
    in_bias = False
    in_combine = False
    in_combine_ker_size = 3
    drop_rate = 0
    bn_size = 8
    bottleneck = False
    out_activation = None
    lr = 5e-5
    weight_decay = 0
    epochs = 500
    model1 = DenseED(in_channels=in_channels, out_channels=out_channels,
                     blocks=blocks,
                     growth_rate=growth_rate,
                     init_features=init_features,
                     in_conv_ker_size=in_conv_ker_size,
                     out_conv_ker_size=out_conv_ker_size,
                     in_dilation=in_dilation,
                     in_batch_norm=in_batch_norm,
                     in_groups=in_groups,
                     in_bias=in_bias,
                     in_combine=in_combine,
                     in_combine_ker_size=in_combine_ker_size,
                     drop_rate=drop_rate,
                     bn_size=bn_size,
                     bottleneck=bottleneck,
                     out_activation=out_activation)


if __name__ == '__main__':

    dense_ed = DenseED(in_channels=10, out_channels=1,
                blocks=(1,1,1,1,1,1,1),
                growth_rate=24,
                init_features=6,
                in_conv_ker_size = 21,
                out_conv_ker_size = 5,
                in_dilation = 1,
                in_batch_norm = True,
                in_groups=1,
                in_bias=False,
                in_combine=False,
                in_combine_ker_size=3,
                drop_rate=0,
                bn_size=8,
                bottleneck=False,
                out_activation=None)

    print(dense_ed)
    x = torch.Tensor(16, 10, 164, 157)
    dense_ed.forward_test(x)
    print(dense_ed._num_parameters_convlayers())
    dense_ed.init_print()
    # print(list(dense_ed.features._modules.items())[14][1])
    # print(dense_ed.features[14]._modules['denselayer1']._modules['conv1'])
    #print(dense_ed.features[6]._modules['conv1'])
    #features = list(dense_ed.features)[:6]
    #print(features)
    #print('**********')
    #print(dense_ed.features[2]._modules['conv2'])
    #rint(dense_ed.features[1].weight.detach().clone().size())
    #ker1 = dense_ed.features[1].weight.detach().clone()[:,None,0,:,:]
    #ker2 = dense_ed.features[1].weight.detach().clone()[:,None,1,:,:]
    #print(ker1)
    #print(ker2)
    #print(ker1.size())
    #print(ker2.size())
    #print('xxxxxxxxxxxx')
    #kernels = dense_ed.features[0].weight.detach().clone()
    #kernels = dense_ed.features[2]._modules['conv2'].weight.detach().clone()
    #kernels = torch.mean(kernels,1,True)
    #kernels = kernels - kernels.min()
    #kernels = kernels / kernels.max()
    #kernels = kernels.reshape(48,1,8*3,6*3)
    #print(kernels.shape)
    #print('+++++++++++')
    #torchvision.utils.save_image(kernels, './image.png')
    #print('**********')
