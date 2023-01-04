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

def SR(x):  # 0000_0000_0111_1111     1111_1111_1000_0000
     x_numpy = x.view(np.uint16)
     sr_result = x_numpy
     sr_flag = 0
     
     is_sr_array         = ((x_numpy < 0x80) & (x_numpy >= 0)) | (x_numpy <= -128)
     first_element_flag  = ((x_numpy[:,0] < 0x40  & (x_numpy[:,0] >=0)) | (x_numpy[:,0] >= -64)).reshape(-1,1)

     row_flag            = (np.sum(is_sr_array,axis=1) == 64).reshape(-1,1)
     sr_flag             = row_flag & first_element_flag
     
     return sr_flag

def ZRLE(x):
     x_numpy = x.view(np.uint16)
     zrle_result = x_numpy
     
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

     return zrle_flag
     
     
def BPC(x):
     # 1024bit (word * 64)
     # print(x[0][0])
     x_numpy = x.view(np.uint16)
     row  = x_numpy.shape[0]
     base_word = x_numpy[:,0].reshape(-1,1)
     delta     = x_numpy[:,1:].reshape(-1,) #- base_word ######################################################## must fix
     delta_uint8    =    delta.view(np.uint8)
     delta_upb      =    np.unpackbits(delta_uint8, bitorder='little')
     delta_upb_2d   =    delta_upb.reshape((row,-1,16))[:,:,::-1]
     dbp_only       =    np.swapaxes(delta_upb_2d, 1, 2)
     dbx            =    dbp_only.copy()
     dbx[:,1:,:]    ^=   dbx[:,:15,:]
     #print(dbx.shape)     # (2,16,63)
     # dbp_only       =     np.array([[[0x0001]*63]*9+[[0x0000]*63]*2+[[0x0001]*63]*5]*2, dtype=np.uint16)                             #2,16,63
     # dbx            =     np.array([[[0x0000]*63]*7+[[0x0001]*63]*1+[[0x0000]*63]*1+[[0x0001]*60+[0x0000]*3]*1+[[0x0000]*63]*1+[[0x0001]*2+[0x0000]*61]*3+[[0x0001]+[0x0000]*62]*1+[[0x0001]*32+[0x0000]*31]*1]
     #                               +[[[0x0001]*32+[0x0000]*31]*16], dtype=np.uint16) # all case     
     dbp_cnt        =    np.sum(dbp_only,axis=2)
     dbx_cnt        =    np.sum(dbx,axis=2) ################################################################################ flag0
     dbxdbp_flag    =    (dbp_cnt == 0) & (dbx_cnt !=0) ## DBX!=0 ,DBP = 0 ################################################# flag1
     dbx_expand     =    np.zeros((dbx.shape[0],dbx.shape[1],dbx.shape[2] + 1), dtype= np.uint8) # 2,16,63+1
     dbx_expand[:,:,1:]  =    dbx
     dbx_pb         =    np.packbits(dbx_expand.reshape(-1,))
     dbx_symbol     =    dbx_pb.view(np.uint64).reshape(2,-1,1)
     two_consec     =    dbx_symbol & (dbx_symbol -1)
     two_consec_result  =    (dbx_symbol == (two_consec + (two_consec >> 1))) & (dbx_symbol != 0)#.reshape((-1,16)) ############################## flag2
     two_consec_flag = two_consec_result.reshape(-1,16)
     # 우선순위로 위에서 부터 결과를 구분짓는 flag를 세우려면..
     # print("dbp : ",dbp_only[0])
     # print("dbx : ",dbx[0])

     ########################## Run length ###############################################
     all_zero         =   (dbx_cnt == 0)
     all_zero_expand_left  =   np.zeros((dbx.shape[0],dbx.shape[1]), dtype= np.uint8)
     all_zero_expand_left[:,1:] = (dbx_cnt[:,:15] == 0)
     
     all_zero_expand_right  =   np.zeros((dbx.shape[0],dbx.shape[1]), dtype= np.uint8)
     all_zero_expand_right[:,:15] = (dbx_cnt[:,1:] == 0)
     
     # print(all_zero)
     # print(all_zero_expand_left)
     # print(all_zero_expand_right)
     # print((all_zero & all_zero_expand_left)) 
     case_1         =    (dbx_cnt == 0) & ((all_zero & all_zero_expand_left) == 0) & ((all_zero & all_zero_expand_right) == 0)     # all zero 1
     case_0         =    (dbx_cnt == 0) & ~case_1      # all zero 2~16

     ###################################################################################
     case_2         =    (dbx_cnt == 63)
     case_3         =    ~case_2 & (dbxdbp_flag == 1)
     case_4         =    ~case_3 & (two_consec_flag == 1)
     case_5         =    ~case_3 & ~case_4 & (dbx_cnt == 1)
     case_6         =    (~case_0) & (~case_1) & (~case_2) & (~case_3) & (~case_4) & (~case_5)
     
     result_flag    =    np.zeros((dbx.shape[0],dbx.shape[1]),dtype=np.uint16)
     result_flag    =    (case_0)*0 + (case_1)*3 + (case_2)*5 + (case_3)*6 + (case_4)*11 + (case_5)*12+ (case_6)*64   
          
     all_zero_2_mask =   np.zeros((dbx.shape[0],dbx.shape[1]+2), dtype= np.uint8)
     all_zero_2_mask[:,1:17] = case_0
     all_zero_2_mask_left = np.zeros((dbx.shape[0],dbx.shape[1]+2), dtype= np.uint8)
     all_zero_2_mask_left[:,1:] = all_zero_2_mask[:,:-1]

     all_zero_2_sum = np.sum((all_zero_2_mask ^ all_zero_2_mask_left),axis=1)//2
     result_sum     =    np.sum(result_flag, axis=1) + all_zero_2_sum*int(6)
     result = (result_sum <= 494)
     # print(result)
     return result   
     

def Comp(x_tensor : torch.Tensor):      # Tensor (128,3,224,224) -> Tensor (64,3,224,224)
    x = x_tensor.numpy().view(np.uint16)
    x_numpy = x.reshape(-1,64) # 94080 16
    
#     print(x_numpy.dtype)
#     # Do SR
     
    sr_flag = SR(x_numpy)
    # Do ZRLE
     
    zrle_flag = ZRLE(x_numpy)
    # Do BPC
     
    # bpc_flag = BPC(x_numpy)

    return sr_flag | zrle_flag | bpc_flag.reshape(-1,1)
    
def Decomp(flag,x_numpy):
     x_numpy_reshape = x_numpy.numpy().reshape(-1,64).view(np.int16)
     # print(x_numpy_reshape.shape) # 8, 64
     # print(flag.shape)            # 8, 8
     result = (x_numpy_reshape & 0xff00) + (x_numpy_reshape & 0x00ff)*flag
     return torch.tensor(result, dtype = torch.float16)
     
# gc.collect()
# torch.cuda.empty_cache()

# model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).to('cuda')
model = fuse_bn_recursively(models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1))#.to('cuda'))
# model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1).to('cuda')
#model = fuse_bn_recursively(models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to('cuda'))
def Quant(x : torch.Tensor, n : int) :

    scale = 0
    zero_p = 0
    # N = 2 ** n
    # N_MIN, N_MAX = -N//2, N//2 - 1
    # x_max, x_min = torch.max(x) , torch.min(x)

    # scale = (x_max - x_min) / (N-1)
    # scale += (x_max * (scale == 0))
    # zero_n = x_max * N_MIN - x_min * N_MAX
    # zero_d = x_max - x_min
    # zero_p =  torch.round(zero_n / (zero_d + 1e-30)) * (zero_d != 0)

    # x_hat = torch.round(x / scale + zero_p)
    # x_q   = torch.clip(x_hat, N_MIN, N_MAX).type(torch.int16)

    return x.half(), scale, zero_p
     
def DeQuant(    x_q: torch.Tensor, 
                scale: torch.Tensor, 
                zero_p: torch.Tensor):
    # return scale  * (x_q - zero_p)
    return x_q.float()

def save_outputs_hook(self, layer_id = str) -> Callable:          
    def fn(_, input) :
        with torch.no_grad():
            Quant_input, scale, zero_p = Quant(input[0],16)
            x = Comp(Quant_input)
            Comp_output = Decomp(x,Quant_input).reshape(Quant_input.shape)
            # print(x)
            # print (Quant_input.shape)
            # print(Comp_output.shape)
            input[0][:] = DeQuant(Comp_output, scale, zero_p).reshape(input[0].shape)
            # input[0][:] = DeQuant(Quant_input, scale, zero_p).reshape(input[0].shape)
            #input[0][:] = DeQuant(torch.tensor(Comp_output), scale, zero_p).reshape(input[0].shape)
            #print(input[0].shape)
    return fn

for name, layer in model.named_modules():
    if ("layer1" != name) | ("layer2" != name) | ("layer3" != name)| ("layer4" != name) :
        layer = dict([*model.named_modules()])[name]
        if isinstance(layer, nn.ReLU):
            layer.register_forward_pre_hook(save_outputs_hook(str(name)))

# for name, param in model.named_parameters():
#     Data_shape = param.shape
#     shape_mul = 1
#     # numpy_Data_shape = param.detach().cpu().numpy()
#     for i in Data_shape:
#         shape_mul *= i
#         # print("i :",i, "shape_mul :",shape_mul)
#     # print ("Data_shape : ",Data_shape, "shape_mul :", shape_mul)
#     # print ("numpy_Data_shape : ",numpy_Data_shape)
#     with torch.no_grad():
#         # print("origin : ",param.view(torch.int16).view(-1))
#         # Quant_input, scale, zero_p = Quant(param,16)
#         # param[:] = DeQuant(Quant_input, scale, zero_p) 
#         if shape_mul%64 == 0 :
#             if ('bn' not in name) :
#                 if ('bias' not in name):
#                     Quant_input, scale, zero_p = Quant(param,16)
#                     x = Comp(Quant_input)
#                     # print(x)
#                     Comp_output = Decomp(x, Quant_input).reshape(Quant_input.shape)
#                     # print("not name : ",name, "x :",x)
#                     # print ("name : ",name,"Dequant.shape :",DeQuant(Comp_output, scale, zero_p).shape, "Data_shape:",Data_shape)
#                     param[:] = DeQuant(Comp_output, scale, zero_p)
#                     # param[:] = DeQuant(Quant_input, scale, zero_p)    
#                 else :
#                     Quant_input, scale, zero_p = Quant(param,16)
#                     param[:] = DeQuant(Quant_input, scale, zero_p)                     
#             else :
#                 Quant_input, scale, zero_p = Quant(param,16)
#                 param[:] = DeQuant(Quant_input, scale, zero_p)    
#         else:
#             Quant_input, scale, zero_p = Quant(param,16)
#             param[:] = DeQuant(Quant_input, scale, zero_p)       
             
#     Data_1d = param.view(-1)
#     # print("Converted : ",Data_1d.view(torch.int16))
     
dataset = dsets.ImageFolder("/media/imagenet/val", models.ResNet50_Weights.IMAGENET1K_V1.transforms()) ### 2번째 인자, transform
loader = DataLoader(dataset= dataset, # dataset
                   batch_size=4,   # batch size power to 2
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
        pred = torch.argmax(output, 1)
        correct += (pred == label).int().sum()
        accum += 4
        #if idx % 20 == 0:
        print(idx, correct /accum * 100, correct, accum)
    acc1 = correct / total * 100

print(acc1)