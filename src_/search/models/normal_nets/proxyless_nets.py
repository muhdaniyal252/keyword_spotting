# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

from modules.layers import *
import json
import librosa
import torchaudio
from convmelspec.stft import ConvertibleSpectrogram as Spectrogram

def proxyless_base(net_config=None, n_classes=1000, bn_param=(0.1, 1e-3), dropout_rate=0):
    assert net_config is not None, 'Please input a network config'
    net_config_path = download_url(net_config)
    net_config_json = json.load(open(net_config_path, 'r'))

    net_config_json['classifier']['out_features'] = n_classes
    net_config_json['classifier']['dropout_rate'] = dropout_rate

    net = ProxylessNASNets.build_from_config(net_config_json)
    net.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

    return net


class MobileInvertedResidualBlock(MyModule):

    def __init__(self, mobile_inverted_conv, shortcut):
        super(MobileInvertedResidualBlock, self).__init__()

        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut

    def forward(self, x):
        if self.mobile_inverted_conv.is_zero_layer():
            res = x
        elif self.shortcut is None or self.shortcut.is_zero_layer():
            res = self.mobile_inverted_conv(x)
        else:
            conv_x = self.mobile_inverted_conv(x)
            skip_x = self.shortcut(x)
            res = skip_x + conv_x
        return res

    @property
    def module_str(self):
        # return '(Conv: %s, Shortcut: %s)' % (
        #     self.mobile_inverted_conv.module_str, self.shortcut.module_str if self.shortcut is not None else None
        # )
        return 'ProxylessNASNets'

    @property
    def config(self):
        return {
            'name': MobileInvertedResidualBlock.__name__,
            'mobile_inverted_conv': self.mobile_inverted_conv.config,
            'shortcut': self.shortcut.config if self.shortcut is not None else None,
        }

    @staticmethod
    def build_from_config(config):
        mobile_inverted_conv = set_layer_from_config(config['mobile_inverted_conv'])
        shortcut = set_layer_from_config(config['shortcut'])
        return MobileInvertedResidualBlock(mobile_inverted_conv, shortcut)

    def get_flops(self, x):
        flops1, conv_x = self.mobile_inverted_conv.get_flops(x)
        if self.shortcut:
            flops2, _ = self.shortcut.get_flops(x)
        else:
            flops2 = 0

        return flops1 + flops2, self.forward(x)

class MFCCModule(nn.Module):
    def __init__(self, n_mfcc=10, hop_length=320, n_fft=640):
        super(MFCCModule, self).__init__()
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.n_fft = n_fft

    def forward(self, x):
        # #Assuming x is the input waveform
        # mfcc = librosa.feature.mfcc(y=x.detach().cpu().numpy(), sr=44100, n_mfcc=self.n_mfcc, hop_length=self.hop_length, n_fft=self.n_fft)
       
        # #Transpose the MFCC matrix to have channels as the first dimension
        # mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32)
        
        # #mfcc_tensor =torch.reshape(mfcc_tensor, (1, mfcc_tensor.shape[0], mfcc_tensor.shape[1]))
        
        # mfcc_tensor = mfcc_tensor.to(x.get_device())
        # return mfcc_tensor

        print(x.get_device())
        melkwargs = {
            'hop_length': self.hop_length,
            'n_fft': self.n_fft,
        }
        mfcc = torchaudio.transforms.MFCC(44100, self.n_mfcc, melkwargs=melkwargs)(x.cpu())
        
        mfcc = mfcc.to(x.device)
        print('mfcc_device:',mfcc.get_device())
        print('mfcc shape: ', mfcc.shape)
        return mfcc

class ProxylessNASNets(MyNetwork):

    def __init__(self, first_conv, blocks, feature_mix_layer, classifier):
        super(ProxylessNASNets, self).__init__()
       
        #self.mfcc_extractor= MFCCModule(n_mfcc=10, hop_length=320, n_fft=640)
        #self.mfcc_extractor= features.mel.MFCC(sr=44100, n_mfcc=10, n_fft=640, hop_length=320, win_length=640)
        self.melspec=Spectrogram(sr=44100, n_fft=640, hop_size=512, n_mel=10)
        self.melspec.set_mode("DFT","store")
        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.feature_mix_layer = feature_mix_layer
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = classifier
        

    def forward(self, x):
        #x = self.mfcc_extractor(x)
        x = self.melspec(x.squeeze(1))
        x = self.first_conv(x.unsqueeze(1))
        for block in self.blocks:
            x = block(x)
        x = self.feature_mix_layer(x)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        # _str = ''
        # for block in self.blocks:
        #     _str += block.unit_str + '\n'
        # return _str
        return 'ProxylessNASNets'


    @property
    def config(self):
        return {
            'name': ProxylessNASNets.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            'feature_mix_layer': self.feature_mix_layer.config,
            'classifier': self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        first_conv = set_layer_from_config(config['first_conv'])
        feature_mix_layer = set_layer_from_config(config['feature_mix_layer'])
        classifier = set_layer_from_config(config['classifier'])
        blocks = []
        for block_config in config['blocks']:
            blocks.append(MobileInvertedResidualBlock.build_from_config(block_config))

        net = ProxylessNASNets(first_conv, blocks, feature_mix_layer, classifier)
        if 'bn' in config:
            net.set_bn_param(**config['bn'])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-3)

        return net

    def get_flops(self, x):
        flop, x = self.first_conv.get_flops(x)

        for block in self.blocks:
            delta_flop, x = block.get_flops(x)
            flop += delta_flop

        delta_flop, x = self.feature_mix_layer.get_flops(x)
        flop += delta_flop

        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatten

        delta_flop, x = self.classifier.get_flops(x)
        flop += delta_flop
        return flop, x
