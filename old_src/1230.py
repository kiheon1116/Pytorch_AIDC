#!/usr/bin/env python
# coding: utf-8

# In[1]:

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

from tqdm import tqdm

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# input : tensor(numpy array)[-1,64] --> output : flag(numpy array)[]
# (block(16word * 64) * n개) => [-1,64], 1개당 int16
def SR(x, length):
     # sign reduction : 1개의 data(2B) 기준 상위 9bit 비교
     x_numpy = x.view(np.int16)
     sr_result = x_numpy
     sr_flag = 0
     ########################### (16:8) ###########################
     # ### 0000_0000_0111_1111 이하,  1111_1111_1000_0000 이상인지. 해당하면 1을 저장.
     is_sr_array         = ((x_numpy < 0x80) & (x_numpy >= 0)) | (x_numpy >= -128)

     # first data(word)에 대한 특수한 조건. (10bit 비교) ### 0000_0000_0011_1111 이하, 1111_1111_1100_0000 이상인지.
     first_element_flag  = ((x_numpy[:,0] < 0x40  & (x_numpy[:,0] >=0)) | (x_numpy[:,0] >= -64)).reshape(-1,1)
     ############################################################

    #  ########################### (16:9) ###########################
    #  # ### 0000_0000_1111_1111 이하,  1111_1111_0000_0000 이상인지. 해당하면 1을 저장.
    #  is_sr_array         = ((x_numpy < 256) & (x_numpy >= 0)) | (x_numpy <= -256)

    #  # first data(word)에 대한 특수한 조건. (10bit 비교) ### 0000_0000_0011_1111 이하, 1111_1111_1100_0000 이상인지.
    #  first_element_flag  = ((x_numpy[:,0] < 0x80  & (x_numpy[:,0] >=0)) | (x_numpy[:,0] <= -128)).reshape(-1,1)
    # #  ############################################################


    #  ########################### (16:10) 1.6:1 ###########################
    #  # ### 0000_0000_0111_1111 이하,  1111_1111_1000_0000 이상인지. 해당하면 1을 저장.
    #  is_sr_array         = ((x_numpy < 512) & (x_numpy >= 0)) | (x_numpy <= -512)

    #  # first data(word)에 대한 특수한 조건. (10bit 비교) ### 0000_0000_0011_1111 이하, 1111_1111_1100_0000 이상인지.
    #  first_element_flag  = ((x_numpy[:,0] < 256  & (x_numpy[:,0] >=0)) | (x_numpy[:,0] <= -256)).reshape(-1,1)
    #  ############################################################


    #  ########################### (16:11) ##########################
    #  # ### 0000_0000_0111_1111 이하,  1111_1111_1000_0000 이상인지. 해당하면 1을 저장.
    #  is_sr_array         = ((x_numpy < 1024) & (x_numpy >= 0)) | (x_numpy <= -1024)

    #  # first data(word)에 대한 특수한 조건. (10bit 비교) ### 0000_0000_0011_1111 이하, 1111_1111_1100_0000 이상인지.
    #  first_element_flag  = ((x_numpy[:,0] < 512  & (x_numpy[:,0] >=0)) | (x_numpy[:,0] <= -512)).reshape(-1,1)
    #  ############################################################

    #  ########################### (16:12) 1.5:1 ###########################
     # ### 0000_0000_0111_1111 이하,  1111_1111_1000_0000 이상인지. 해당하면 1을 저장.
    #  is_sr_array         = ((x_numpy < 2048) & (x_numpy >= 0)) | (x_numpy <= -2048)

    #  # first data(word)에 대한 특수한 조건. (10bit 비교) ### 0000_0000_0011_1111 이하, 1111_1111_1100_0000 이상인지.
    #  first_element_flag  = ((x_numpy[:,0] < 1024  & (x_numpy[:,0] >=0)) | (x_numpy[:,0] <= -1024)).reshape(-1,1)
    #  ############################################################

    #  ########################### (16:13) ###########################
    #  # ### 0000_0000_0111_1111 이하,  1111_1111_1000_0000 이상인지. 해당하면 1을 저장.
    #  is_sr_array         = ((x_numpy < 4096) & (x_numpy >= 0)) | (x_numpy <= -4096)

    #  # first data(word)에 대한 특수한 조건. (10bit 비교) ### 0000_0000_0011_1111 이하, 1111_1111_1100_0000 이상인지.
    #  first_element_flag  = ((x_numpy[:,0] < 2048  & (x_numpy[:,0] >=0)) | (x_numpy[:,0] <= -2048)).reshape(-1,1)
    #  ############################################################

    #  ########################### (16:14) ###########################
    #  # ### 0000_0000_0111_1111 이하,  1111_1111_1000_0000 이상인지. 해당하면 1을 저장.
    #  is_sr_array         = ((x_numpy < 8192) & (x_numpy >= 0)) | (x_numpy <= -8192)

    #  # first data(word)에 대한 특수한 조건. (10bit 비교) ### 0000_0000_0011_1111 이하, 1111_1111_1100_0000 이상인지.
    #  first_element_flag  = ((x_numpy[:,0] < 4096  & (x_numpy[:,0] >=0)) | (x_numpy[:,0] <= -4096)).reshape(-1,1)
    # #  ############################################################

    #  ########################### (16:15) ###########################
    #  # ### 0000_0000_0111_1111 이하,  1111_1111_1000_0000 이상인지. 해당하면 1을 저장.
    #  is_sr_array         = ((x_numpy < 16384) & (x_numpy >= 0)) | (x_numpy <= -16384)

    #  # first data(word)에 대한 특수한 조건. (10bit 비교) ### 0000_0000_0011_1111 이하, 1111_1111_1100_0000 이상인지.
    #  first_element_flag  = ((x_numpy[:,0] < 8192  & (x_numpy[:,0] >=0)) | (x_numpy[:,0] <= -8192)).reshape(-1,1)
    #  ############################################################

     # is_sr_array[-1,64] 일 때, 모든 2차원 (word 단위의 64개)가 sign reduction 가능한지.
     row_flag            = (np.sum(is_sr_array, axis=1) == length).reshape(-1,1) # length = 64
     sr_flag             = row_flag & first_element_flag

     return sr_flag

def ZRLE(x, length):
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

     zrle_flag            = (np.sum(non_zero_size,axis=1) < ((length/2)*16)-2).reshape(-1,1) # length 64기준 510  # 510 574 638 702 766 830 894 958

     return zrle_flag

def BPC(input, length):
     input_np = input.view(np.int16)
     row = input_np.shape[0]

     # Data Block
     input_np_base = input_np[:,0].reshape(-1,1)
     base = input_np_base.copy()
     base_uint8 = base.view(np.uint8)
     base_unpackbits = np.unpackbits(base_uint8, bitorder = 'little')
     BASE = base_unpackbits.reshape((row,16))[:,::-1]

     # delta
     input_np_delta = input_np[:,1:].reshape(-1,length-1) - input_np_base # 64기준 length - 1 = 63

     # Delta-BP
     delta_uint8 = input_np_delta.view(np.uint8) # 현재 little endian
     delta_unpackbits = np.unpackbits(delta_uint8, bitorder ='little')
     delta_unpackbits_2d = delta_unpackbits.reshape((row,length-1,16))[:,:,::-1] # 64기준 length-1 = 63
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
     DBP_XOR_SYMBOL = DBP_XOR_packbits.view(np.uint64).reshape(row,-1,1) ######################################################################## length 변경시 uint(length)로 변경해주어야함.

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
     case_2 = (DBP_XOR_CNT == length-1) # 64기준 length - 1 = 63

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
     result = (result_sum <= ((length/2)*16-18)) # 64기준 length*16-18 = 512-18
     # 494,  558, 622, 686, 750, 814, 878, 942
     # 8bit -> 2:1(512), 7bit -> 16:9(576), 6bit -> 16:10(640),  5bit -> 16:11(704),
     # 4bit -> 16:12(768), 3bit -> 16:13(832), 2bit -> 16:14(896), 1bit -> 16:15(960)

     return result

def Comp(x_tensor, length):      # Tensor (128,3,224,224) -> Tensor (64,3,224,224)
    x = x_tensor.cpu().numpy().view(np.int16)
    x_numpy = x.reshape(-1,length) # 94080 16
    sr_flag = SR(x_numpy,length)
    # Do ZRLE

    zrle_flag = ZRLE(x_numpy,length)
    # Do BPC

    bpc_flag = BPC(x_numpy,length)

    # print('sr_flag : ', np.sum(sr_flag.astype(np.int32)) / len(sr_flag.reshape(-1,)))
    # print('zrle_flag : ', np.sum(zrle_flag.astype(np.int32)) / len(zrle_flag.reshape(-1,)))
    # print('bpc_flag : ', np.sum(bpc_flag.astype(np.int32)) / len(bpc_flag.reshape(-1,)))
    return sr_flag | zrle_flag | bpc_flag.reshape(-1,1)

def rolling(x_torch,flag_torch,length,dtype):
    lossy_result = (x_torch.reshape(-1,length)*(~flag_torch) + (flag_torch)*1)
    index = torch.FloatTensor(x_torch.shape).type(torch.int16)
    rolling_zero = (~flag_torch)&(lossy_result == 0x0000)    
    rolling_first = (~flag_torch)&(((lossy_result > 0x0000) & (lossy_result < 2**(dtype))) | ((lossy_result <= -1) & (lossy_result >= -(2**dtype))))#.view(np.int16)
    # index += rolling_first*1
    for i in range(16-(dtype+1)):
        print(i)
        rolling = (~flag_torch)&(((lossy_result > 2**(dtype+i)) & (lossy_result < 2**(dtype+i+1))) | ((lossy_result <= -(2**(dtype+i)+1)) & (lossy_result >= -(2**(dtype+i+1)))))#.view(np.int16)
        # index += rolling*i
    rolling_last = (~flag_torch)&((lossy_result == 32767) | (lossy_result == -32768) )#.view(np.int16)
    # index += rolling_last
    
    # # rolling 결과들 dtype == bool
    # rolling = (~flag_torch)&(lossy_result == 0x0000)    
    # rolling_0 = (~flag_torch)&(((lossy_result > 0x0000) & (lossy_result < 128)) | ((lossy_result <= -1) & (lossy_result >= -128)))#.view(np.int16)
    # rolling_1 = (~flag_torch)&(((lossy_result >= 128) & (lossy_result < 256)) | ((lossy_result <= -129) & (lossy_result >= -256)))#.view(np.int16)
    # rolling_2 = (~flag_torch)&(((lossy_result >= 256) & (lossy_result < 512)) | ((lossy_result <= -257) & (lossy_result >= -512)))#.view(np.int16)
    # rolling_3 = (~flag_torch)&(((lossy_result >= 512) & (lossy_result < 1024)) | ((lossy_result <= -513) & (lossy_result >= -1024)))#.view(np.int16)
    # rolling_4 = (~flag_torch)&(((lossy_result >= 1024) & (lossy_result < 2048)) | ((lossy_result <= -1025) & (lossy_result >= -2048)))#.view(np.int16)
    # rolling_5 = (~flag_torch)&(((lossy_result >= 2048) & (lossy_result < 4096)) | ((lossy_result <= -2049) & (lossy_result >= -4096)))#.view(np.int16)
    # rolling_6 = (~flag_torch)&(((lossy_result >= 4096) & (lossy_result < 8192)) | ((lossy_result <= -4097) & (lossy_result >= -8192)))#.view(np.int16)
    # rolling_7 = (~flag_torch)&(((lossy_result >= 8192) & (lossy_result < 16384)) | ((lossy_result <= -8193) & (lossy_result >= -16384)))#.view(np.int16)
    # rolling_8 = (~flag_torch)&(((lossy_result >= 16384) & (lossy_result < 32767)) | ((lossy_result <= -16385) & (lossy_result >= -32768)))#.view(np.int16)
    # rolling_9 = (~flag_torch)&((lossy_result == 32767) | (lossy_result == -32768) )#.view(np.int16)
    
    # INT7_index = rolling_0*0+rolling_1*1+rolling_2*2+rolling_3*3+rolling_4*4+rolling_5*5+rolling_6*6+rolling_7*7+rolling_8*8+rolling_9*9
    # index = (dtype == 7) & rolling_first&0x007f+rolling[0]&0x00fe+rolling[1]&0x01fc+rolling[2]&0x03f8+rolling[3]&0x07f0+rolling[4]&0x0fe0+rolling[5]&0x1fc0+rolling[6]&0x3f80+rolling[7]&0x7f00+rolling_last&0xfe00
    # index = rolling_first&0x00ff+rolling[0]&0x01fe+rolling[1]&0x03fc+rolling[2]&0x07f8+rolling[3]&0x0ff0+rolling[4]&0x1fe0+rolling[5]&0x3fc0+rolling[6]&0x7f80+rolling[7]&0xff00
    # INT9 = rolling_first&0x01ff+rolling[0]&0x03fe+rolling[1]&0x07fc+rolling[2]&0x0ff8+rolling[3]&0x1ff0+rolling[4]&0x3fe0+rolling[5]&0x7fc0+rolling[6]&0xff80
    # INT10 = rolling_first&0x03ff+rolling[0]&0x07fe+rolling[1]&0x0ffc+rolling[2]&0x1ff8+rolling[3]&0x3ff0+rolling[4]&0x7fe0+rolling[5]&0xffc0
    # INT11 = rolling_first&0x07ff+rolling[0]&0x0ffe+rolling[1]&0x1ffc+rolling[2]&0x3ff8+rolling[3]&0x7ff0+rolling[4]&0xffe0
    # INT12 = rolling_first&0x0fff+rolling[0]&0x1ffe+rolling[1]&0x3ffc+rolling[2]&0x7ff8+rolling[3]&0xfff0
    # INT13 = rolling_first&0x1fff+rolling[0]&0x3ffe+rolling[1]&0x7ffc+rolling[2]&0xfff8
    # INT14 = rolling_first&0x3fff+rolling[0]&0x7ffe+rolling[1]&0xfffc
    # INT15 = rolling_first&0x7fff+rolling[0]&0xfffe
    return index.type(torch.int16)

# lossy
def rolling_index(x_torch,flag_torch,length,dtype):
    lossy_result = (x_torch.reshape(-1,length)*(~flag_torch) + (flag_torch)*1)
    
    # rolling 결과들 dtype == bool
    # 양수
    rolling = (~flag_torch)&(lossy_result == 0x0000)
    rolling_0 = (~flag_torch)&((lossy_result > 0x0000) & (lossy_result < 64)) # 0000_0000_0000_0000 ~ 0000_0000_0011_1111
    rolling_1 = (~flag_torch)&((lossy_result >= 64) & (lossy_result < 128))   # 0000_0000_0100_0000 ~ 0000_0000_0111_1111
    rolling_2 = (~flag_torch)&((lossy_result >= 128) & (lossy_result < 256))  # 0000 0000 1000 0000
    rolling_3 = (~flag_torch)&((lossy_result >= 256) & (lossy_result < 512))  # 0000 0001 0000 0000
    rolling_4 = (~flag_torch)&((lossy_result >= 512) & (lossy_result < 1024)) # 0000 0010
    rolling_5 = (~flag_torch)&((lossy_result >= 1024) & (lossy_result < 2048))# 0000 0100
    rolling_6 = (~flag_torch)&((lossy_result >= 2048) & (lossy_result < 4096))# 0000 1000
    rolling_7 = (~flag_torch)&((lossy_result >= 4096) & (lossy_result < 8192))# 0001 0000
    rolling_8 = (~flag_torch)&((lossy_result >= 8192) & (lossy_result < 16384)) # 0010 0000
    rolling_9 = (~flag_torch)&((lossy_result >= 16384) & (lossy_result < 32767)) # 0100 0000
    # 음수
    rolling_m0 = (~flag_torch)&((lossy_result <= -1) & (lossy_result >= -64))#.view(np.int16)               # 1111 1111 1100 0000 ~
    rolling_m1 = (~flag_torch)&((lossy_result <= -65) & (lossy_result >= -128))#.view(np.int16)             # 1111 1111 1100 0000 ~
    rolling_m2 = (~flag_torch)&((lossy_result <= -129) & (lossy_result >= -256))#.view(np.int16)            # 1111 1111 1000 0000 ~ 
    rolling_m3 = (~flag_torch)&((lossy_result <= -257) & (lossy_result >= -512))#.view(np.int16)            # 1111 1111 0000 0000 ~ 
    rolling_m4 = (~flag_torch)&((lossy_result <= -513) & (lossy_result >= -1024))#.view(np.int16)
    rolling_m5 = (~flag_torch)&((lossy_result <= -1025) & (lossy_result >= -2048))#.view(np.int16)
    rolling_m6 = (~flag_torch)&((lossy_result <= -2049) & (lossy_result >= -4096))#.view(np.int16)
    rolling_m7 = (~flag_torch)&((lossy_result <= -4097) & (lossy_result >= -8192))#.view(np.int16)
    rolling_m8 = (~flag_torch)&((lossy_result <= -8193) & (lossy_result >= -16384))#.view(np.int16)
    rolling_m9 = (~flag_torch)&((lossy_result <= -16385) & (lossy_result >= -32768))#.view(np.int16)
    # rolling_m9 = (~flag_torch)&(lossy_result == -32768) #.view(np.int16)
        
    if(dtype==7):
        index = ((rolling*11)+(rolling_0*1)+(rolling_1*2)+(rolling_2*3)+(rolling_3*4)+(rolling_4*5)+(rolling_5*6)+(rolling_6*7)+(rolling_7*8)+(rolling_8*9)+(rolling_9*10)+(rolling_m0*-1)+(rolling_m1*-2)+(rolling_m2*-3)+(rolling_m3*-4)+(rolling_m4*-5)+(rolling_m5*-6)+(rolling_m6*-7)+(rolling_m7*-8)+(rolling_m8*-9)+(rolling_m9*-10))
    elif(dtype==8):
        index = ((rolling*11)+((rolling_0|rolling_1)*1)+(rolling_2*2)+(rolling_3*3)+(rolling_4*4)+(rolling_5*5)+(rolling_6*6)+(rolling_7*7)+(rolling_8*8)+(rolling_9*9)+((rolling_m0|rolling_m1)*-1)+(rolling_m2*-2)+(rolling_m3*-3)+(rolling_m4*-4)+(rolling_m5*-5)+(rolling_m6*-6)+(rolling_m7*-7)+(rolling_m8*-8)+(rolling_m9*-9))
    elif(dtype==9):
        index = ((rolling*11)+((rolling_0|rolling_1|rolling_2)*1)+(rolling_3*2)+(rolling_4*3)+(rolling_5*4)+(rolling_6*5)+(rolling_7*6)+(rolling_8*7)+(rolling_9*8)+((rolling_m0|rolling_m1|rolling_m2)*-1)+(rolling_m3*-2)+(rolling_m4*-3)+(rolling_m5*-4)+(rolling_m6*-5)+(rolling_m7*-6)+(rolling_m8*-7)+(rolling_m9*-8))
    elif(dtype==10):
         index =((rolling*11)+((rolling_0|rolling_1|rolling_2|rolling_3)*1)+(rolling_4*2)+(rolling_5*3)+(rolling_6*4)+(rolling_7*5)+(rolling_8*6)+(rolling_9*7)+((rolling_m0|rolling_m1|rolling_m2|rolling_m3)*-1)+(rolling_m4*-2)+(rolling_m5*-3)+(rolling_m6*-4)+(rolling_m7*-5)+(rolling_m8*-6)+(rolling_m9*-7))
    elif(dtype==11):
        index = ((rolling*11)+((rolling_0|rolling_1|rolling_2|rolling_3|rolling_4)*1)+(rolling_5*2)+(rolling_6*3)+(rolling_7*4)+(rolling_8*5)+(rolling_9*6)+((rolling_m0|rolling_m1|rolling_m2|rolling_m3|rolling_m4)*-1)+(rolling_m5*-2)+(rolling_m6*-3)+(rolling_m7*-4)+(rolling_m8*-5)+(rolling_m9*-6))
    elif(dtype==12):
        index = ((rolling*11)+((rolling_0|rolling_1|rolling_2|rolling_3|rolling_4|rolling_5)*1)+(rolling_6*2)+(rolling_7*3)+(rolling_8*4)+(rolling_9*5)+((rolling_m0|rolling_m1|rolling_m2|rolling_m3|rolling_m4|rolling_m5)*-1)+(rolling_m6*-2)+(rolling_m7*-3)+(rolling_m8*-4)+(rolling_m9*-5))
    elif(dtype==13):
        index = ((rolling*11)+((rolling_0|rolling_1|rolling_2|rolling_3|rolling_4|rolling_5|rolling_6)*1)+(rolling_7*2)+(rolling_8*3)+(rolling_9*4)+((rolling_m0|rolling_m1|rolling_m2|rolling_m3|rolling_m4|rolling_m5|rolling_m6)*-1)+(rolling_m7*-2)+(rolling_m8*-3)+(rolling_m9*-4))
    elif(dtype==14):
        index = ((rolling*11)+((rolling_0|rolling_1|rolling_2|rolling_3|rolling_4|rolling_5|rolling_6|rolling_7)*1)+(rolling_8*2)+(rolling_9*3)+((rolling_m0|rolling_m1|rolling_m2|rolling_m3|rolling_m4|rolling_m5|rolling_m6|rolling_m7)*-1)+(rolling_m8*-2)+(rolling_m9*-3))
    elif(dtype==15):
        index = ((rolling*11)+((rolling_0|rolling_1|rolling_2|rolling_3|rolling_4|rolling_5|rolling_6|rolling_7|rolling_8)*1)+(rolling_9*2)+((rolling_m0|rolling_m1|rolling_m2|rolling_m3|rolling_m4|rolling_m5|rolling_m6|rolling_m7|rolling_m8)*-1)+(rolling_m9*-2))
    return index.type(torch.int16)

# lossy + lossless
def rolling_total(x_torch,length,dtype):
    lossy_result = (x_torch.reshape(-1,length))
    
    # rolling 결과들 dtype == bool
    # 양수
    rolling = (lossy_result == 0x0000)
    rolling_0 = ((lossy_result > 0x0000) & (lossy_result < 64)) # 0000_0000_0000_0000 ~ 0000_0000_0011_1111
    rolling_1 = ((lossy_result >= 64) & (lossy_result < 128))   # 0000_0000_0100_0000 ~ 0000_0000_0111_1111
    rolling_2 = ((lossy_result >= 128) & (lossy_result < 256))  # 0000 0000 1000 0000 ~ 0000_0000_1111_1111
    rolling_3 = ((lossy_result >= 256) & (lossy_result < 512))  # 0000 0001 0000 0000 ~ 0000_0001_1111_1111
    rolling_4 = ((lossy_result >= 512) & (lossy_result < 1024)) # 0000 0010
    rolling_5 = ((lossy_result >= 1024) & (lossy_result < 2048))# 0000 0100
    rolling_6 = ((lossy_result >= 2048) & (lossy_result < 4096))# 0000 1000
    rolling_7 = ((lossy_result >= 4096) & (lossy_result < 8192))# 0001 0000
    rolling_8 = ((lossy_result >= 8192) & (lossy_result < 16384)) # 0010 0000
    rolling_9 = ((lossy_result >= 16384) & (lossy_result < 32767)) # 0100 0000
    # 음수
    rolling_m0 = ((lossy_result <= -1) & (lossy_result >= -64))#.view(np.int16)               # 1111 1111 1100 0000 ~
    rolling_m1 = ((lossy_result <= -65) & (lossy_result >= -128))#.view(np.int16)             # 1111 1111 1100 0000 ~
    rolling_m2 = ((lossy_result <= -129) & (lossy_result >= -256))#.view(np.int16)            # 1111 1111 1000 0000 ~ 
    rolling_m3 = ((lossy_result <= -257) & (lossy_result >= -512))#.view(np.int16)            # 1111 1111 0000 0000 ~ 
    rolling_m4 = ((lossy_result <= -513) & (lossy_result >= -1024))#.view(np.int16)
    rolling_m5 = ((lossy_result <= -1025) & (lossy_result >= -2048))#.view(np.int16)
    rolling_m6 = ((lossy_result <= -2049) & (lossy_result >= -4096))#.view(np.int16)
    rolling_m7 = ((lossy_result <= -4097) & (lossy_result >= -8192))#.view(np.int16)
    rolling_m8 = ((lossy_result <= -8193) & (lossy_result >= -16384))#.view(np.int16)
    rolling_m9 = ((lossy_result <= -16385) & (lossy_result >= -32768))#.view(np.int16)
    # rolling_m9 = (~flag_torch)&(lossy_result == -32768) #.view(np.int16)
        
    if(dtype==7):
        index = ((rolling*11)+(rolling_0*1)+(rolling_1*2)+(rolling_2*3)+(rolling_3*4)+(rolling_4*5)+(rolling_5*6)+(rolling_6*7)+(rolling_7*8)+(rolling_8*9)+(rolling_9*10)+(rolling_m0*-1)+(rolling_m1*-2)+(rolling_m2*-3)+(rolling_m3*-4)+(rolling_m4*-5)+(rolling_m5*-6)+(rolling_m6*-7)+(rolling_m7*-8)+(rolling_m8*-9)+(rolling_m9*-10))
    elif(dtype==8):
        index = ((rolling*11)+((rolling_0|rolling_1)*1)+(rolling_2*2)+(rolling_3*3)+(rolling_4*4)+(rolling_5*5)+(rolling_6*6)+(rolling_7*7)+(rolling_8*8)+(rolling_9*9)+((rolling_m0|rolling_m1)*-1)+(rolling_m2*-2)+(rolling_m3*-3)+(rolling_m4*-4)+(rolling_m5*-5)+(rolling_m6*-6)+(rolling_m7*-7)+(rolling_m8*-8)+(rolling_m9*-9))
    elif(dtype==9):
        index = ((rolling*11)+((rolling_0|rolling_1|rolling_2)*1)+(rolling_3*2)+(rolling_4*3)+(rolling_5*4)+(rolling_6*5)+(rolling_7*6)+(rolling_8*7)+(rolling_9*8)+((rolling_m0|rolling_m1|rolling_m2)*-1)+(rolling_m3*-2)+(rolling_m4*-3)+(rolling_m5*-4)+(rolling_m6*-5)+(rolling_m7*-6)+(rolling_m8*-7)+(rolling_m9*-8))
    elif(dtype==10):
         index =((rolling*11)+((rolling_0|rolling_1|rolling_2|rolling_3)*1)+(rolling_4*2)+(rolling_5*3)+(rolling_6*4)+(rolling_7*5)+(rolling_8*6)+(rolling_9*7)+((rolling_m0|rolling_m1|rolling_m2|rolling_m3)*-1)+(rolling_m4*-2)+(rolling_m5*-3)+(rolling_m6*-4)+(rolling_m7*-5)+(rolling_m8*-6)+(rolling_m9*-7))
    elif(dtype==11):
        index = ((rolling*11)+((rolling_0|rolling_1|rolling_2|rolling_3|rolling_4)*1)+(rolling_5*2)+(rolling_6*3)+(rolling_7*4)+(rolling_8*5)+(rolling_9*6)+((rolling_m0|rolling_m1|rolling_m2|rolling_m3|rolling_m4)*-1)+(rolling_m5*-2)+(rolling_m6*-3)+(rolling_m7*-4)+(rolling_m8*-5)+(rolling_m9*-6))
    elif(dtype==12):
        index = ((rolling*11)+((rolling_0|rolling_1|rolling_2|rolling_3|rolling_4|rolling_5)*1)+(rolling_6*2)+(rolling_7*3)+(rolling_8*4)+(rolling_9*5)+((rolling_m0|rolling_m1|rolling_m2|rolling_m3|rolling_m4|rolling_m5)*-1)+(rolling_m6*-2)+(rolling_m7*-3)+(rolling_m8*-4)+(rolling_m9*-5))
    elif(dtype==13):
        index = ((rolling*11)+((rolling_0|rolling_1|rolling_2|rolling_3|rolling_4|rolling_5|rolling_6)*1)+(rolling_7*2)+(rolling_8*3)+(rolling_9*4)+((rolling_m0|rolling_m1|rolling_m2|rolling_m3|rolling_m4|rolling_m5|rolling_m6)*-1)+(rolling_m7*-2)+(rolling_m8*-3)+(rolling_m9*-4))
    elif(dtype==14):
        index = ((rolling*11)+((rolling_0|rolling_1|rolling_2|rolling_3|rolling_4|rolling_5|rolling_6|rolling_7)*1)+(rolling_8*2)+(rolling_9*3)+((rolling_m0|rolling_m1|rolling_m2|rolling_m3|rolling_m4|rolling_m5|rolling_m6|rolling_m7)*-1)+(rolling_m8*-2)+(rolling_m9*-3))
    elif(dtype==15):
        index = ((rolling*11)+((rolling_0|rolling_1|rolling_2|rolling_3|rolling_4|rolling_5|rolling_6|rolling_7|rolling_8)*1)+(rolling_9*2)+((rolling_m0|rolling_m1|rolling_m2|rolling_m3|rolling_m4|rolling_m5|rolling_m6|rolling_m7|rolling_m8)*-1)+(rolling_m9*-2))
    return index.type(torch.int16)


def shift(x_torch,  length, count, dtype):
    # lossless + lossy
    shift_index_per_word = rolling_total(x_torch,length,dtype)  # (-1,32) torch.int16
    # lossy
    # shift_index_per_word = rolling_index(x_torch,flag_torch,length,dtype)  # (-1,32) torch.int16

    shift_shape = shift_index_per_word.shape #(-1,32) int16
    shift_index_per_64 = shift_index_per_word.reshape(-1,int(length/count),count) # (-1,8,4) torch.int16  
    shift_max = torch.max(shift_index_per_64,dim=2) #(-1,8) int16
    shift_min = torch.min(shift_index_per_64,dim=2) #(-1,8) int16
    shift_index_per_64_max_abs = torch.abs(shift_max[0])  
    shift_index_per_64_min_abs = torch.abs(shift_min[0])
    ge_compare_flag = torch.ge(shift_index_per_64_min_abs,shift_index_per_64_max_abs)
    shift = ((ge_compare_flag*shift_min[0]) + ((~ge_compare_flag)*shift_max[0])).reshape(-1,1)
    shift_index = shift.expand(-1,count).reshape(shift_shape) # (-1,32) int16    
    return shift_index    

def shift_to_mask(x_torch, dtype):
    x_shape = x_torch.shape
    
    dtype_filter    = torch.arange(10) < (17 - dtype)
    mask            = torch.tensor([-(1 << i) for i in range(10)]) * dtype_filter
    offset          = torch.tensor([(256 >> (9 - i)) for i in range(10)]) * dtype_filter
    
    mask_flag = torch.zeros((10, x_shape[0], x_shape[1]), dtype=torch.int16)
    for idx in range(10):
        mask_flag[idx] = (torch.abs(x_torch) == (idx + 1))

    mask_block = torch.sum(mask_flag * mask.reshape(-1,1,1), dim=0)
    offset_block = torch.sum(mask_flag * offset.reshape(-1,1,1), dim=0)

    return mask_block.type(torch.int16).cuda(), offset_block.type(torch.int16).cuda()
    
    # # 1000
    # if(dtype==7):
    #     mask_block = ((mask_m1*0xffff)+(mask_m2*0xffff)+(mask_m3*0xfffe)+(mask_m4*0xfffc)+(mask_m5*0xfff8)+(mask_m6*0xfff0)+(mask_m7*0xffe0)+(mask_m8*0xffc0)+(mask_m9*0xff10)+(mask_m10*0xff00))
    # elif(dtype==8):
    #     mask_block = ((mask_m1*0xffff)+(mask_m2*0xffff)+(mask_m3*0xfffe)+(mask_m4*0xfffc)+(mask_m5*0xfff8)+(mask_m6*0xfff0)+(mask_m7*0xffe0)+(mask_m8*0xffc0)+(mask_m9*0xff10))
    # elif(dtype==9):
    #     mask_block = ((mask_m1*0xffff)+(mask_m2*0xffff)+(mask_m3*0xfffe)+(mask_m4*0xfffc)+(mask_m5*0xfff8)+(mask_m6*0xfff0)+(mask_m7*0xffe0)+(mask_m8*0xffc0))
    # elif(dtype==10):
    #     mask_block = ((mask_m1*0xffff)+(mask_m2*0xffff)+(mask_m3*0xfffe)+(mask_m4*0xfffc)+(mask_m5*0xfff8)+(mask_m6*0xfff0)+(mask_m7*0xffe0))
    # elif(dtype==11):
    #     mask_block = ((mask_m1*0xffff)+(mask_m2*0xffff)+(mask_m3*0xfffe)+(mask_m4*0xfffc)+(mask_m5*0xfff8)+(mask_m6*0xfff0))
    # elif(dtype==12):
    #     mask_block = ((mask_m1*0xffff)+(mask_m2*0xffff)+(mask_m3*0xfffe)+(mask_m4*0xfffc)+(mask_m5*0xfff8))
    # elif(dtype==13):
    #     mask_block = ((mask_m1*0xffff)+(mask_m2*0xffff)+(mask_m3*0xfffe)+(mask_m4*0xfffc))
    # elif(dtype==14):
    #     mask_block = ((mask_m1*0xffff)+(mask_m2*0xffff)+(mask_m3*0xfffe))
    # elif(dtype==15):
    #     mask_block = ((mask_m1*0xffff)+(mask_m2*0xffff))
    
    # return mask_block.type(torch.int16)  

# mask = [0x80,0xC0,0xE0,0xF0,0xF8,0xFC,0xFE,0xFF]
def Decomp(x_numpy, flag, mask_block, length):
     x_numpy_reshape = x_numpy.cpu().numpy().reshape(-1,length).view(np.int16)
     mask_block_numpy = mask_block.numpy()
     result = ((x_numpy_reshape & mask_block_numpy)*(~flag)) + ((x_numpy_reshape & 0xffff)*flag)
     return torch.tensor(result.astype(np.int16))


model = fuse_bn_recursively(models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)).to('cuda')
# model = fuse_bn_recursively(models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1))#.to('cuda'))
# model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1).to('cuda')
#model = fuse_bn_recursively(models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to('cuda'))

def Quant(x, n) :

    N = 2 ** n
    N_MIN, N_MAX = -N//2 + 1 , N//2 - 1
    x_max, x_min = torch.max(x) , torch.min(x)
    # x_max_abs = torch.abs(x_max)
    # x_min_abs = torch.abs(x_min)
    # x_abs_flag = torch.ge(x_max_abs,x_min_abs)
    # x_max = x_abs_flag*x_max_abs + (~x_abs_flag)*x_min_abs
    # x_min = -x_max

    scale = (x_max - x_min) / (N-2)
    scale += (x_max * (scale == 0))
    zero_n = x_max * N_MIN - x_min * N_MAX
    zero_d = x_max - x_min
    zero_p =  torch.round(zero_n / (zero_d + 1e-30)) * (zero_d != 0)

    x_hat = torch.round(x / scale + zero_p)
    x_q   = torch.clip(x_hat, N_MIN, N_MAX).type(torch.int16)

    return x_q, scale, zero_p

def DeQuant(    x_q,
                scale,
                zero_p):
    return scale  * (x_q.to('cuda') - zero_p)

def quantize_channel(x, n):

    N = 2 ** n
    N_MIN, N_MAX = -N//2 +1, N//2 - 1

    if len(x.shape) >= 4:
        x_2d  = x.view(x.shape[0], -1)
        x_max = torch.max(x_2d,dim=1)[0]
        x_min = torch.min(x_2d,dim=1)[0]
        x_max_abs = torch.abs(x_max)
        x_min_abs = torch.abs(x_min)
        x_abs_flag = torch.ge(x_max_abs,x_min_abs)

        x_max = x_abs_flag*x_max_abs + (~x_abs_flag)*x_min_abs
        x_min = -x_max

        scale = ((x_max - x_min) / (N-2))
        scale += (x_max * (scale == 0))
        scale = scale.view(x.shape[0], -1)

        zero_n = x_max * N_MIN - x_min * N_MAX
        zero_d = x_max - x_min
        zero_p = torch.round(zero_n / (zero_d + 1e-30)) * (zero_d != 0)
        zero_p = zero_p.view(x.shape[0], -1)

        x_hat = torch.round(x_2d / scale + zero_p)
        x_q   = torch.clip(x_hat, N_MIN, N_MAX).view(x.shape).type(torch.int16)
        return x_q, scale, zero_p

    x_max, x_min = torch.abs(torch.max(x)) , torch.abs(torch.min(x))
    x_max_abs = torch.abs(x_max)
    x_min_abs = torch.abs(x_min)
    x_abs_flag = torch.ge(x_max_abs,x_min_abs)
    x_max = x_abs_flag*x_max_abs + (~x_abs_flag)*x_min_abs
    x_min = -x_max

    scale = (x_max - x_min) / (N-2)
    scale += (x_max * (scale == 0))
    zero_n = x_max * N_MIN - x_min * N_MAX
    zero_d = x_max - x_min
    zero_p = torch.round(zero_n / (zero_d + 1e-30)) * (zero_d != 0)

    x_hat = torch.round(x / scale + zero_p)
    x_q   = torch.clip(x_hat, N_MIN, N_MAX).type(torch.int16)

    return x_q, scale, zero_p

def dequantize_channel  (   x_q,
                            scale,
                            zero_p):

    x_2d = x_q.view(x_q.shape[0], -1).to('cuda')
    return (scale  * (x_2d - zero_p)).view(x_q.shape)

def save_outputs_hook(layer_id) -> Callable:
    def fn(_, input) :
        with torch.no_grad():
            pass
            # Quant_input, scale, zero_p = Quant(input[0],16)#quantize_channel(input[0],16)
            # flag = Comp(Quant_input,length)
            # Comp_output = Decomp(flag, Quant_input,length).reshape(Quant_input.shape)
            # input[0][:] = DeQuant(Comp_output, scale, zero_p)#dequantize_channel(Comp_output, scale, zero_p)

            # Quant_input, scale, zero_p = Quant(input[0],16)
            # flag = Comp(Quant_input)
            # Comp_output = Decomp(flag,Quant_input).reshape(Quant_input.shape)
            # input[0][:] = DeQuant(Comp_output, scale, zero_p).reshape(input[0].shape)
            # print(layer_id, 'fmap', np.sum(flag) / len(flag.reshape(-1,)))
            # input[0][:] = DeQuant(Quant_input, scale, zero_p).reshape(input[0].shape)
            # input[0][:] = DeQuant(torch.tensor(Comp_output), scale, zero_p).reshape(input[0].shape)
    return fn

for name, layer in model.named_modules():
    if ("layer1" != name) | ("layer2" != name) | ("layer3" != name)| ("layer4" != name) :
        layer = dict([*model.named_modules()])[name]
        #if isinstance(layer, nn.ReLU):
        # layer.register_forward_pre_hook(save_outputs_hook(name))

length = 64
count = 1
dtype = 7
parameters_index = np.array([]).astype(np.int16)
for name, param in tqdm(model.named_parameters()):
    Data_shape = param.shape
    shape_mul = 1
    for i in Data_shape:
        shape_mul *= i
    with torch.no_grad():
        if shape_mul%64 != 0 :
            Quant_input, scale, zero_p = Quant(param,16)#quantize_channel(param,16)
            param[:] = DeQuant(Quant_input, scale, zero_p)#dequantize_channel(Comp_output, scale, zero_p)
            continue

        if 'bn' in name:
            Quant_input, scale, zero_p = Quant(param,16)#quantize_channel(param,16)
            param[:] = DeQuant(Quant_input, scale, zero_p)#dequantize_channel(Comp_output, scale, zero_p)
            continue

        if 'bias' in name:
            Quant_input, scale, zero_p = Quant(param,16)#quantize_channel(param,16)
            param[:] = DeQuant(Quant_input, scale, zero_p)#dequantize_channel(Comp_output, scale, zero_p)
            continue
        
        # Step 1 : Quantization
        Quant_input, scale, zero_p = Quant(param,16)#quantize_channel(param,16)         # input (294*32) INT16
        Quant_input_shape = Quant_input.shape
        # print("input",Quant_input[0][1:5], Quant_input.shape, Quant_input.dtype)

        # Step 2 : Compression 
        # flag = Comp(Quant_input,length)        # flag (294*1) bool
        # flag_torch = torch.from_numpy(flag).to('cuda')

        # Step 3 : rolling (shift)
        shift_block = shift(Quant_input,length,count,dtype)        # shift (294*32) 4bit

        mask_block, offset_block = shift_to_mask(shift_block,dtype)        # mask (294*32) INT16 
        # print(mask_block.is_cuda, offset_block.is_cuda, Quant_input.is_cuda)
        Quant_input = (Quant_input.reshape(mask_block.shape) & mask_block ).reshape(Quant_input_shape)#+ offset_block
               
        # Step 4 : Decompression
        # Data_1 [(294*32)*INT7] +  Data_2 [(294)*32bit]
        # Comp_output = Decomp(Quant_input, flag, mask_block, length).reshape(Quant_input.shape)
        # print("output",Comp_output[0][1:5])
        
        #@ Test 1 : total param & param per layer
        # param_index = torch.abs(rolling_total(Quant_input, length, dtype))
        # param_save = np.save("/home/kkh/pytorch/data/param_data/"+ name,param_index.cpu().numpy())
        # parameters_index = np.append(parameters_index.reshape(-1,).tolist(),param_index.tolist()).astype(np.int16)
        
        # #@ Test 2 : total comp_data & comp_data per layer
        # param_comp_index = torch.abs(rolling_total(Comp_output.to('cuda'), length, dtype))
        # param_save = np.save("/home/kkh/pytorch/data/param_comp_data/"+ name,param_comp_index.cpu().numpy())
        # parameters_comp_index = np.append(parameters_index.reshape(-1,).tolist(),param_comp_index.tolist()).astype(np.int16)

        # Step 5 : DeQuantization
        param[:] = DeQuant(Quant_input, scale, zero_p)#dequantize_channel(Comp_output, scale, zero_p)

        #@ Test print
        # print(name, param.shape, param.max(), param.min(), np.sum(flag.astype(np.int32)) / len(flag.reshape(-1,)))
        
#@ Test 2 : total param
# print("total",parameters_index.shape, parameters_index.dtype,"\n")
# param_save = np.save("/home/kkh/pytorch/data/param_data/total.weight",parameters_index)
# param_save = np.save("/home/kkh/pytorch/data/param_comp_data/total.weight",parameters_comp_index)

dataset = dsets.ImageFolder("/media/imagenet/val", models.ResNet50_Weights.IMAGENET1K_V1.transforms()) ### 2번째 인자, transform
loader = DataLoader(dataset= dataset, # dataset
                  batch_size= 128,   # batch size power to 2
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
       input = input.cuda(non_blocking=True)
       label = label.cuda(non_blocking=True)
       output = model(input)
       # print(output)
       pred = torch.argmax(output, 1)
       correct += (pred == label).int().sum()
       accum += 4
       if idx % 1000 == 0:
           print(idx, correct /accum * 100, correct, accum)
    acc1 = correct / total * 100

print(acc1)
# %%
