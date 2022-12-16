import numpy as np 
import torch, gc
import torchvision.models as models
import torchvision.datasets as dsets
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Iterable, Callable
from bn_fold import fuse_bn_recursively


from tqdm import tqdm 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cudnn.enabled = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
import torch
from tqdm import tqdm


np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# input : tensor(numpy array)[-1,64] --> output : flag(numpy array)[]
# (block(16word * 64) * n개) => [-1,64], 1개당 int16
def SR(x):  
     # sign reduction : 1개의 data(2B) 기준 상위 9bit 비교  
     x_numpy = x.view(np.int16)
     sr_result = x_numpy
     sr_flag = 0
     ########################### (16:8) ###########################     
     # ### 0000_0000_0111_1111 이하,  1111_1111_1000_0000 이상인지. 해당하면 1을 저장.
     is_sr_array         = ((x_numpy < 0x80) & (x_numpy >= 0)) | (x_numpy <= -128)
     
     # first data(word)에 대한 특수한 조건. (10bit 비교) ### 0000_0000_0011_1111 이하, 1111_1111_1100_0000 이상인지.
     first_element_flag  = ((x_numpy[:,0] < 0x40  & (x_numpy[:,0] >=0)) | (x_numpy[:,0] >= -64)).reshape(-1,1)
     ############################################################

    #  ########################### (16:9) ###########################     
    #  # ### 0000_0000_1111_1111 이하,  1111_1111_0000_0000 이상인지. 해당하면 1을 저장.
    #  is_sr_array         = ((x_numpy < 256) & (x_numpy >= 0)) | (x_numpy <= -256)
     
    #  # first data(word)에 대한 특수한 조건. (10bit 비교) ### 0000_0000_0011_1111 이하, 1111_1111_1100_0000 이상인지.
    #  first_element_flag  = ((x_numpy[:,0] < 0x80  & (x_numpy[:,0] >=0)) | (x_numpy[:,0] >= -128)).reshape(-1,1)
    # #  ############################################################


    #  ########################### (16:10) 1.6:1 ###########################     
    #  # ### 0000_0000_0111_1111 이하,  1111_1111_1000_0000 이상인지. 해당하면 1을 저장.
    #  is_sr_array         = ((x_numpy < 512) & (x_numpy >= 0)) | (x_numpy <= -512)
     
    #  # first data(word)에 대한 특수한 조건. (10bit 비교) ### 0000_0000_0011_1111 이하, 1111_1111_1100_0000 이상인지.
    #  first_element_flag  = ((x_numpy[:,0] < 256  & (x_numpy[:,0] >=0)) | (x_numpy[:,0] >= -256)).reshape(-1,1)
    #  ############################################################


    #  ########################### (16:11) ##########################     
    #  # ### 0000_0000_0111_1111 이하,  1111_1111_1000_0000 이상인지. 해당하면 1을 저장.
    #  is_sr_array         = ((x_numpy < 1024) & (x_numpy >= 0)) | (x_numpy <= -1024)
     
    #  # first data(word)에 대한 특수한 조건. (10bit 비교) ### 0000_0000_0011_1111 이하, 1111_1111_1100_0000 이상인지.
    #  first_element_flag  = ((x_numpy[:,0] < 512  & (x_numpy[:,0] >=0)) | (x_numpy[:,0] >= -512)).reshape(-1,1)
    #  ############################################################

    #  ########################### (16:12) 1.5:1 ###########################     
     # ### 0000_0000_0111_1111 이하,  1111_1111_1000_0000 이상인지. 해당하면 1을 저장.
    #  is_sr_array         = ((x_numpy < 2048) & (x_numpy >= 0)) | (x_numpy <= -2048)
     
    #  # first data(word)에 대한 특수한 조건. (10bit 비교) ### 0000_0000_0011_1111 이하, 1111_1111_1100_0000 이상인지.
    #  first_element_flag  = ((x_numpy[:,0] < 1024  & (x_numpy[:,0] >=0)) | (x_numpy[:,0] >= -1024)).reshape(-1,1)
    #  ############################################################

    #  ########################### (16:13) ###########################     
    #  # ### 0000_0000_0111_1111 이하,  1111_1111_1000_0000 이상인지. 해당하면 1을 저장.
    #  is_sr_array         = ((x_numpy < 4096) & (x_numpy >= 0)) | (x_numpy <= -4096)
     
    #  # first data(word)에 대한 특수한 조건. (10bit 비교) ### 0000_0000_0011_1111 이하, 1111_1111_1100_0000 이상인지.
    #  first_element_flag  = ((x_numpy[:,0] < 2048  & (x_numpy[:,0] >=0)) | (x_numpy[:,0] >= -2048)).reshape(-1,1)
    #  ############################################################

    #  ########################### (16:14) ###########################     
    #  # ### 0000_0000_0111_1111 이하,  1111_1111_1000_0000 이상인지. 해당하면 1을 저장.
    #  is_sr_array         = ((x_numpy < 8192) & (x_numpy >= 0)) | (x_numpy <= -8192)
     
    #  # first data(word)에 대한 특수한 조건. (10bit 비교) ### 0000_0000_0011_1111 이하, 1111_1111_1100_0000 이상인지.
    #  first_element_flag  = ((x_numpy[:,0] < 4096  & (x_numpy[:,0] >=0)) | (x_numpy[:,0] >= -4096)).reshape(-1,1)
    # #  ############################################################

    #  ########################### (16:15) ###########################     
    #  # ### 0000_0000_0111_1111 이하,  1111_1111_1000_0000 이상인지. 해당하면 1을 저장.
    #  is_sr_array         = ((x_numpy < 16384) & (x_numpy >= 0)) | (x_numpy <= -16384)
     
    #  # first data(word)에 대한 특수한 조건. (10bit 비교) ### 0000_0000_0011_1111 이하, 1111_1111_1100_0000 이상인지.
    #  first_element_flag  = ((x_numpy[:,0] < 8192  & (x_numpy[:,0] >=0)) | (x_numpy[:,0] >= -8192)).reshape(-1,1)
    #  ############################################################ 

     # is_sr_array[-1,64] 일 때, 모든 2차원 (word 단위의 64개)가 sign reduction 가능한지. 
     row_flag            = (np.sum(is_sr_array, axis=1) == 64).reshape(-1,1)
     sr_flag             = row_flag & first_element_flag
     
     return sr_flag

def ZRLE(x): 
     # 1개의 word(2B) 기준 단위로 zero, non_zero 파악 이후 encoding
     x_numpy = x.view(np.int16)
    
     # word 단위(int16)로 non_zero의 경우 flag = 1인 tensor
     # 즉, tensor는 Byte 단위이므로, int16 -> 1B로 표현해서
     # 4 word를 묶어 64bit 단위로 pattern을 파악하기 위해서는 4B씩 묶어야 하므로 
     # 기존의 1B씩 된 mask를 int32로 표현하여 4개씩 묶음.
     non_zero_element         = (x_numpy != 0)
     
     non_zero_element_pattern = non_zero_element.view(np.int32)
     
     non_zero_size      = (non_zero_element_pattern == 0x00000000) * 6
     
     non_zero_size     += (non_zero_element_pattern == 0x01000000) * 22
     non_zero_size     += (non_zero_element_pattern == 0x00010000) * 21
     non_zero_size     += (non_zero_element_pattern == 0x00000100) * 21
     non_zero_size     += (non_zero_element_pattern == 0x00000001) * 21
     
     non_zero_size     += (non_zero_element_pattern == 0x01010000) * 36
     non_zero_size     += (non_zero_element_pattern == 0x01000100) * 36
     non_zero_size     += (non_zero_element_pattern == 0x01000001) * 36
     non_zero_size     += (non_zero_element_pattern == 0x00010100) * 36
     non_zero_size     += (non_zero_element_pattern == 0x00010001) * 36
     non_zero_size     += (non_zero_element_pattern == 0x00000101) * 36
     
     non_zero_size     += (non_zero_element_pattern == 0x01010100) * 52
     non_zero_size     += (non_zero_element_pattern == 0x01010001) * 52
     non_zero_size     += (non_zero_element_pattern == 0x01000101) * 52
     non_zero_size     += (non_zero_element_pattern == 0x00010101) * 52
     
     non_zero_size     += (non_zero_element_pattern == 0x01010101) * 66

     zrle_flag            = (np.sum(non_zero_size,axis=1) < 510).reshape(-1,1)
     # 510 574 638 702 766 830 894 958

     return zrle_flag
     
def BPC(input):
     input_np = input.view(np.int16)
     row = input_np.shape[0]
     
     # Data Block
     input_np_base = input_np[:,0].reshape(-1,1)
     base = input_np_base.copy()
     base_uint8 = base.view(np.uint8)
     base_unpackbits = np.unpackbits(base_uint8, bitorder = 'little')
     BASE = base_unpackbits.reshape((row,16))[:,::-1]
     
     # delta
     input_np_delta = input_np[:,1:].reshape(-1,63) - input_np_base
     
     
     # Delta-BP
     delta_uint8 = input_np_delta.view(np.uint8) # 현재 little endian
     delta_unpackbits = np.unpackbits(delta_uint8, bitorder ='little')
     delta_unpackbits_2d = delta_unpackbits.reshape((row,63,16))[:,:,::-1]
     Delta_BP = np.swapaxes(delta_unpackbits_2d, 1, 2) # (row,16,63)
     Delta_BP_CNT = np.sum(Delta_BP, axis=2) # (row,16)
     
     
     # DBP_XOR
     DBP_XOR = Delta_BP.copy()
     DBP_XOR[:,1:,:] ^= DBP_XOR[:,:15,:] # (row,16,63)
     DBP_XOR_CNT = np.sum(DBP_XOR, axis=2) # (row,16)
     DBP_XOR_SUM = np.zeros((DBP_XOR.shape[0],DBP_XOR.shape[1],DBP_XOR.shape[2]+1), dtype = np.uint8)
     DBP_XOR_SUM[:,:,0] = BASE
     DBP_XOR_SUM[:,:,1:] = DBP_XOR
     DBP_XOR_packbits =  np.packbits(DBP_XOR_SUM.reshape(-1,))
     DBP_XOR_SYMBOL = DBP_XOR_packbits.view(np.uint64).reshape(row,-1,1)
     
     
     
     # Encoding
          
     ## Case 1 : All-0's 
     mask_all_zero  = (DBP_XOR_CNT == 0)
     mask_all_zero_left  = np.zeros(mask_all_zero.shape, dtype=np.uint8)
     mask_all_zero_right = np.zeros(mask_all_zero.shape, dtype=np.uint8)     
     mask_all_zero_left[:,:15] = (DBP_XOR_CNT[:,1:] == 0)
     mask_all_zero_left[:,1:] = (DBP_XOR_CNT[:,:15] == 0)
     case_1 = (DBP_XOR_CNT == 0) & ((mask_all_zero & mask_all_zero_left) == 0) & ((mask_all_zero & mask_all_zero_right) == 0)
     ### All-0's run length 2~16
     case_0 = (DBP_XOR_CNT == 0) & ~case_1
     
     ## Case 2 : All-1's 
     case_2 = (DBP_XOR_CNT == 63)
     
     ## Case 3 : DBP_XOR != 0 & Delta_BP == 0
     dbx_dbp_flag = (DBP_XOR_CNT != 0) & (Delta_BP_CNT ==0)
     case_3 = ~case_2 & (dbx_dbp_flag == 1)
     
     ## Case 4 : Consecutive two 1's 
     mask_two_consec = DBP_XOR_SYMBOL & (DBP_XOR_SYMBOL-1)
     mask_two_consec_result = (DBP_XOR_CNT == 2).reshape(row,16,1) & (DBP_XOR_SYMBOL == (mask_two_consec + (mask_two_consec >> 1))) 
     mask_two_consec_flag = mask_two_consec_result.reshape(-1,16)
     case_4 = ~case_2 & ~case_3 & (mask_two_consec_flag == 1)

     ## Case 5 : Single 1
     case_5 = ~case_2 & ~case_3 & ~case_4 & (DBP_XOR_CNT == 1)
     
     ## Case 6 : Uncompressed
     case_6 = (~case_0) & (~case_1) & (~case_2) & (~case_3) & (~case_4) & (~case_5)

     result_flag = np.zeros((DBP_XOR.shape),dtype=np.int16)
     # result_flag = (case_0)*0 + (case_1)*3 + (case_2)*5 + (case_3)*5 + (case_4)*11 + (case_5)*11+ (case_6)*64   
     result_flag = (case_0)*0 + (case_1)*3 + (case_2)*5 + (case_3)*6 + (case_4)*11 + (case_5)*12+ (case_6)*64   

     mask_all_zero_2 = np.zeros((DBP_XOR.shape[0],DBP_XOR.shape[1]+2), dtype=np.uint8)
     mask_all_zero_2_left = np.zeros((DBP_XOR.shape[0],DBP_XOR.shape[1]+2), dtype=np.uint8)
     mask_all_zero_2[:,1:17] = case_0
     mask_all_zero_2_left[:,:17] = mask_all_zero_2[:,1:]
     mask_all_zero_2_sum = np.sum((mask_all_zero_2 ^ mask_all_zero_2_left), axis=1)//2
     
     result_sum = np.sum(result_flag, axis=1) + mask_all_zero_2_sum*int(6)
     result = (result_sum <= 494)  # 494,  558, 622, 686, 750, 814, 878, 942
     # 8bit -> 2:1(512), 7bit -> 16:9(576), 6bit -> 16:10(640),  5bit -> 16:11(704), 
     # 4bit -> 16:12(768), 3bit -> 16:13(832), 2bit -> 16:14(896), 1bit -> 16:15(960)
     
     return result

     

def Comp(x_tensor : torch.Tensor):      # Tensor (128,3,224,224) -> Tensor (64,3,224,224)
    x = x_tensor.numpy().view(np.int16)
    x_numpy = x.reshape(-1,64) # 94080 16
    
#     print(x_numpy.dtype)
#     # Do SR
     
    sr_flag = SR(x_numpy)
    # Do ZRLE
     
    zrle_flag = ZRLE(x_numpy)
    # Do BPC
     
    bpc_flag = BPC(x_numpy)

    return sr_flag | zrle_flag | bpc_flag.reshape(-1,1)


# mask = [0x80,0xC0,0xE0,0xF0,0xF8,0xFC,0xFE,0xFF]
def Decomp(flag,x_numpy):
     x_numpy_reshape = x_numpy.numpy().reshape(-1,64).view(np.int16)
     # print(x_numpy_reshape.shape) # 8, 64
     # print(flag.shape)            # 8, 8
     result = (x_numpy_reshape & 0xff00) + (x_numpy_reshape & 0x00ff)*flag
     return torch.tensor(result.astype(np.int16))
     
# gc.collect()
# torch.cuda.empty_cache()

model = fuse_bn_recursively(models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1))#.to('cuda')
# model = fuse_bn_recursively(models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1))#.to('cuda'))
# model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1).to('cuda')
#model = fuse_bn_recursively(models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to('cuda'))

############################################ for FP#######################################
# def Quant(x : torch.Tensor, n : int) :

#     scale = 0
#     zero_p = 0
#     N = 2 ** n
#     N_MIN, N_MAX = -N//2, N//2 - 1
#     x_max, x_min = torch.max(x) , torch.min(x)

#     scale = (x_max - x_min) / (N-1)
#     scale += (x_max * (scale == 0))
#     zero_n = x_max * N_MIN - x_min * N_MAX
#     zero_d = x_max - x_min
#     zero_p =  torch.round(zero_n / (zero_d + 1e-30)) * (zero_d != 0)

#     x_hat = torch.round(x / scale + zero_p)
#     x_q   = torch.clip(x_hat, N_MIN, N_MAX).type(torch.int16)

#     return x.half(), scale, zero_p
     
# def DeQuant(    x_q: torch.Tensor, 
#                 scale: torch.Tensor, 
#                 zero_p: torch.Tensor):
#     # return scale  * (x_q - zero_p)
#     return x_q.float()
#########################################################################################
def Quant(x : torch.Tensor, n : int) :
     
    N = 2 ** n
    N_MIN, N_MAX = -N//2, N//2 - 1
    x_max, x_min = torch.max(x) , torch.min(x)

    scale = (x_max - x_min) / (N-1)
    scale += (x_max * (scale == 0))
    zero_n = x_max * N_MIN - x_min * N_MAX
    zero_d = x_max - x_min
    zero_p =  torch.round(zero_n / (zero_d + 1e-30)) * (zero_d != 0)

    x_hat = torch.round(x / scale + zero_p)
    x_q   = torch.clip(x_hat, N_MIN, N_MAX).type(torch.int16)

    return x_q, scale, zero_p
     
def DeQuant(    x_q: torch.Tensor, 
                scale: torch.Tensor, 
                zero_p: torch.Tensor):
    return scale  * (x_q - zero_p)

def quantize_channel(x: torch.Tensor, n : int):
     
    N = 2 ** n
    N_MIN, N_MAX = -N//2, N//2 - 1

    if len(x.shape) >= 4:
        x_2d  = x.view(x.shape[0], -1)
        x_max = torch.max(x_2d,dim=1)[0]
        x_min = torch.min(x_2d,dim=1)[0]
        print("x_max",x_max.shape)
        print("x_min",x_min.shape)
        x_max_abs = torch.abs(torch.max(x_2d,dim=1)[0])
        x_min_abs = torch.abs(torch.min(x_2d,dim=1)[0])
        print("x_max_abs",x_max_abs.shape)
        print("x_min_abs",x_min_abs[:])
        if(x_max[:][0] >= x_min[:][0]): x_min[:] = - x_max[:] # 22.12.16 modifying
        else : x_max[:] = x_min[:]

        scale = ((x_max - x_min) / (N-1))
        scale += (x_max * (scale == 0))
        scale = scale.view(x.shape[0], -1)

        zero_n = x_max * N_MIN - x_min * N_MAX
        zero_d = x_max - x_min
        zero_p = torch.round(zero_n / (zero_d + 1e-30)) * (zero_d != 0)
        zero_p = zero_p.view(x.shape[0], -1)

        x_hat = torch.round(x_2d / scale + zero_p)
        x_q   = torch.clip(x_hat, N_MIN, N_MAX).view(x.shape).type(torch.int8)
        return x_q, scale, zero_p

    x_max, x_min = torch.abs(torch.max(x)) , torch.abs(torch.min(x))
    if(x_max >= x_min): x_min = - x_max
    else : x_max = x_min

    scale = (x_max - x_min) / (N-1)
    scale += (x_max * (scale == 0))
    zero_n = x_max * N_MIN - x_min * N_MAX
    zero_d = x_max - x_min
    zero_p = torch.round(zero_n / (zero_d + 1e-30)) * (zero_d != 0)

    x_hat = torch.round(x / scale + zero_p)
    x_q   = torch.clip(x_hat, N_MIN, N_MAX).type(torch.int8)

    return x_q, scale, zero_p

def dequantize_channel  (   x_q: torch.Tensor, 
                            scale: torch.Tensor, 
                            zero_p: torch.Tensor):

    x_2d = x_q.view(x_q.shape[0], -1)
    return (scale  * (x_2d - zero_p)).view(x_q.shape)

def save_outputs_hook(layer_id) -> Callable:          
    def fn(_, input) :
        with torch.no_grad():
            Quant_input, scale, zero_p = Quant(input[0],16)
            flag = Comp(Quant_input)
            Comp_output = Decomp(flag,Quant_input).reshape(Quant_input.shape)
            input[0][:] = DeQuant(Comp_output, scale, zero_p).reshape(input[0].shape)
            # print(layer_id, 'fmap', np.sum(flag) / len(flag.reshape(-1,)))
            # # print(layer_id, input[0][:])
            # input[0][:] = DeQuant(Quant_input, scale, zero_p).reshape(input[0].shape)
            # input[0][:] = DeQuant(torch.tensor(Comp_output), scale, zero_p).reshape(input[0].shape)
            #print(input[0].shape)
    return fn

for name, layer in model.named_modules():
    if ("layer1" != name) | ("layer2" != name) | ("layer3" != name)| ("layer4" != name) :
        layer = dict([*model.named_modules()])[name]
        #if isinstance(layer, nn.ReLU):
        # layer.register_forward_pre_hook(save_outputs_hook(name))

for name, param in model.named_parameters():
    Data_shape = param.shape
    shape_mul = 1
    # numpy_Data_shape = param.detach().cpu().numpy()
    for i in Data_shape:
        shape_mul *= i
    with torch.no_grad():
        # print("origin : ",param.view(torch.int16).view(-1))
        # Quant_input, scale, zero_p = Quant(param,16)
        # param[:] = DeQuant(Quant_input, scale, zero_p) 
        if shape_mul%64 != 0 :
            Quant_input, scale, zero_p = quantize_channel(param,16)
            param[:] = dequantize_channel(Quant_input, scale, zero_p)
            continue

        if 'bn' in name:
            Quant_input, scale, zero_p = quantize_channel(param,16)
            param[:] = dequantize_channel(Quant_input, scale, zero_p)
            continue
        
        if 'bias' in name:
            Quant_input, scale, zero_p = quantize_channel(param,16)
            param[:] = dequantize_channel(Quant_input, scale, zero_p)
            continue    

        # Quant_input, scale, zero_p = Quant(param,16)
        # param[:] = DeQuant(Quant_input, scale, zero_p)
                    
        Quant_input, scale, zero_p = quantize_channel(param,16)
        flag = Comp(Quant_input)
        # print(name, np.sum(flag) / len(flag.reshape(-1,)))
        Comp_output = Decomp(flag, Quant_input).reshape(Quant_input.shape)
        param[:] = dequantize_channel(Comp_output, scale, zero_p)
     
dataset = dsets.ImageFolder("/media/imagenet/val", models.ResNet50_Weights.IMAGENET1K_V1.transforms()) ### 2번째 인자, transform
loader = DataLoader(dataset= dataset, # dataset
                   batch_size= 4,   # batch size power to 2
                   shuffle = False, # false
                   num_workers = 8, # num_workers 
                   pin_memory=True) # pin_memory 

correct = 0
total = 50000
accum = 0
model.eval()
# torch.no_grad()
with torch.no_grad():
    for idx, (input, label) in enumerate(tqdm(loader)):
        input = input#.cuda(non_blocking=True)
        label = label#.cuda(non_blocking=True)     
        output = model(input)    
        # print(output)
        pred = torch.argmax(output, 1)
        correct += (pred == label).int().sum()
        accum += 4
        if idx % 1000 == 0:
            print(idx, correct /accum * 100, correct, accum)
    acc1 = correct / total * 100

print(acc1)