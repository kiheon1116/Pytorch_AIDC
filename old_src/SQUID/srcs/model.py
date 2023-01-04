import torch

from torch.utils.data import DataLoader 

import torchvision.models as models
import torchvision.datasets as dsets 

import galois

from tqdm import tqdm

from inject_error import *
from bn_fold import fuse_bn_recursively

class Model():
    def __init__(self, args):
        self.args = args
        self.weight = self.get_weight(args)
        self.model  = self.make_model(args)
        self.loader = self.make_loader(args)
        
    def validate(self):
        self.model = self.model.to('cuda')
        self.model.eval()
        correct = 0
        total = len(self.loader.iterable.dataset)
        with torch.no_grad():
            for img, label in self.loader:
                img = img.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)
                out = self.model(img)
                pred = torch.argmax(out, 1)
                correct += (pred == label).int().sum()
        acc1 = correct / total * 100
        return acc1.item()

    def get_weight(self, args):
        weight_name = self.get_weight_name(args.model)
        return getattr(getattr(models, weight_name), args.weight)

    def get_weight_name(self, model_name):
        model_name_dict = { 
            'alexnet'           : 'AlexNet',                
            'convnext_base'     : 'ConvNeXt_Base',
            'convnext_large'    : 'ConvNeXt_Large',
            'convnext_small'    : 'ConvNeXt_Small',
            'convnext_tiny'     : 'ConvNeXt_Tiny',
            'densenet121'       : 'DenseNet121',
            'densenet161'       : 'DenseNet161',
            'densenet169'       : 'DenseNet169',
            'densenet201'       : 'DenseNet201',
            'efficientnet_b0'   : 'EfficientNet_B0',
            'efficientnet_b1'   : 'EfficientNet_B1',
            'efficientnet_b2'   : 'EfficientNet_B2',
            'efficientnet_b3'   : 'EfficientNet_B3',
            'efficientnet_b4'   : 'EfficientNet_B4',
            'efficientnet_b5'   : 'EfficientNet_B5',
            'efficientnet_b6'   : 'EfficientNet_B6',
            'efficientnet_b7'   : 'EfficientNet_B7',
            'googlenet'         : 'GoogLeNet',
            'inception_v3'      : 'Inception_V3',
            'mnasnet0_5'        : 'MNASNet0_5',
            'mnasnet1_0'        : 'MNASNet1_0',
            'mobilenet_v2'      : 'MobileNet_V2',
            'mobilenet_v3_large': 'MobileNet_V3_Large',
            'mobilenet_v3_small': 'MobileNet_V3_Small',
            'regnet_x_16gf'     : 'RegNet_X_16GF',
            'regnet_x_1_6gf'    : 'RegNet_X_1_6GF',
            'regnet_x_32gf'     : 'RegNet_X_32GF',
            'regnet_x_3_2gf'    : 'RegNet_X_3_2GF',
            'regnet_x_400mf'    : 'RegNet_X_400MF',
            'regnet_x_800mf'    : 'RegNet_X_800MF',
            'regnet_x_8gf'      : 'RegNet_X_8GF',
            'regnet_y_16gf'     : 'RegNet_Y_16GF',
            'regnet_y_1_6gf'    : 'RegNet_Y_1_6GF',
            'regnet_y_32gf'     : 'RegNet_Y_32GF',
            'regnet_y_3_2gf'    : 'RegNet_Y_3_2GF',
            'regnet_y_400mf'    : 'RegNet_Y_400MF',
            'regnet_y_800mf'    : 'RegNet_Y_800MF',
            'regnet_y_8gf'      : 'RegNet_Y_8GF',
            'resnext101_32x8d'  : 'ResNeXt101_32X8D',
            'resnext50_32x4d'   : 'ResNeXt50_32X4D',
            'resnet101'         : 'ResNet101',
            'resnet152'         : 'ResNet152',
            'resnet18'          : 'ResNet18',
            'resnet34'          : 'ResNet34',
            'resnet50'          : 'ResNet50',
            'shufflenet_v2_x0_5': 'ShuffleNet_V2_X0_5',
            'shufflenet_v2_x1_0': 'ShuffleNet_V2_X1_0',
            'squeezenet1_0'     : 'SqueezeNet1_0',
            'squeezenet1_1'     : 'SqueezeNet1_1',
            'vgg11'             : 'VGG11',
            'vgg11_bn'          : 'VGG11_BN',
            'vgg13'             : 'VGG13',
            'vgg13_bn'          : 'VGG13_BN',
            'vgg16'             : 'VGG16',
            'vgg16_bn'          : 'VGG16_BN',
            'vgg19'             : 'VGG19',
            'vgg19_bn'          : 'VGG19_BN',
            'vit_b_16'          : 'ViT_B_16',
            'vit_b_32'          : 'ViT_B_32',
            'vit_l_16'          : 'ViT_L_16',
            'vit_l_32'          : 'ViT_L_32',
            'wide_resnet50_2'   : 'Wide_ResNet50_2',
            'wide_resnet101_2'  : 'Wide_ResNet101_2'
        }

        return model_name_dict[model_name] + '_Weights'
    
    def make_model(self, args):
        model = getattr(models, args.model)(weights=self.weight)
        if args.option == 'weight_nulling' or args.option == 'no_error':
            model.half()
        if args.quant == 'channel':
            return fuse_bn_recursively(model)
        return model

    def make_loader(self, args):
        dset = dsets.ImageFolder(args.dset_path, self.weight.transforms())
        loader = DataLoader(dataset = dset, 
                            batch_size = args.batch,
                            shuffle = False,
                            num_workers = args.num_workers,
                            pin_memory = True)
        desc = self.make_loader_desc(args)
        bar_format = self.make_loader_bar_format()
        return tqdm(loader, desc=desc, bar_format = bar_format)

    def make_loader_desc(self, args):
        desc = f'<{args.model}_int{args.nbit}({args.quant})_{args.option}'
        desc += f'|seed{args.l_seed % 10000:04d}|{args.ber:.1f}>'
        return desc

    def make_loader_bar_format(self):
        return '{desc}|{bar:5}|{percentage:5.1f}% [{elapsed}<{remaining}]'

class ErrorModel(Model):
    def __init__(self, args):
        super(ErrorModel, self).__init__(args)
        self.p = 10 ** args.ber
        self.encode_lut = self.get_encode_lut(args)
        self.decode_lut = self.get_decode_lut(args)
        if args.option == 'no_error' or args.option == 'default':
            self.inject_error_to_param_default(args)
            return
        if args.option == 'weight_nulling':
            self.inject_error_to_param_and_recovery(args)
            self.model.float()
            return
        self.inject_error_to_param_and_recovery(args)
        return

    def get_encode_lut(self, args):
        if args.nbit == 7:
            return np.array([0,15,8,4,2,1,0,0], dtype=np.uint8)
        if args.nbit == 6:
            return np.array([0,0,15,8,4,2,1,0], dtype=np.uint8)
        if args.nbit == 5:
            return np.array([0,0,0,15,8,4,2,1], dtype=np.uint8)

    def get_decode_lut(self, args):            
        if args.nbit == 7:
            return np.array([    0, 4, 8,12,16,20,24,96, 
                                32,36,40,80,48,72,68,64], dtype=np.uint8)
        if args.nbit == 6:
            return np.array([    0, 2, 4, 6, 8,10,12,48, 
                                16,18,20,40,24,36,34,32], dtype=np.uint8)
        if args.nbit == 5:
            return np.array([    0, 1, 2, 3, 4, 5, 6,24, 
                                 8, 9,10,20,12,18,17,16], dtype=np.uint8)        

    def inject_error_to_param_default(self, args):
        is_ch_quant = (args.quant == 'channel')
        for name, param in self.model.named_parameters():
            if not param.shape:
                continue
            if 'bias' in name:
                continue
            if args.option == 'no_error':
                inject_no_err(param, args.nbit, is_ch_quant)
                continue
            if args.option == 'default':
                inject_err_default(param, self.p, args.nbit, is_ch_quant)
                continue

    def inject_error_to_param_and_recovery(self, args):
        bch = galois.BCH(127, 113)
        rs  = galois.ReedSolomon(15, 2 * args.nbit - 1)
        mask = 2 ** args.nbit - 1
        is_ch_quant = (args.quant == 'channel')
        for name, param in self.model.named_parameters():
            if not param.shape:
                continue
            if 'bias' in name:
                continue
            if args.option == 'bch':
                inject_err_bch( param, self.p, args.nbit,
                                bch, mask, is_ch_quant)
                continue
            if args.option == 'vapi':
                inject_err_vapi(   param, self.p, args.nbit,
                                bch, mask, is_ch_quant)
                continue
            if args.option == 'squid':
                inject_err_squid(   param, self.p, args.nbit, 
                                    self.encode_lut, self.decode_lut,
                                    rs, mask, is_ch_quant)
                continue
            if args.option == 'multi_bch':
                inject_multi_err_bch(   param, self.p, args.nbit,
                                bch, mask, is_ch_quant)
                continue    
            if args.option == 'multi_vapi':
                inject_multi_err_vapi(   param, self.p, args.nbit,
                                bch, mask, is_ch_quant)
                continue
            if args.option == 'multi_squid':
                inject_multi_err_squid(   param, self.p, args.nbit, 
                                self.encode_lut, self.decode_lut,
                                rs, mask, is_ch_quant)
                continue
            if args.option == 'weight_nulling':
                inject_err_weight_null(param, self.p, args.nbit, mask)
                continue
        return 