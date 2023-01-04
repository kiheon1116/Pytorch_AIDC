import numpy as np
import math
import time
import sys

np.set_printoptions(threshold=np.inf, linewidth=np.inf)
test = np.load('/users/user2/Desktop/AIDC/model/features.npy').reshape(-1, 64)
# print(test.shape)

# data = test[383]
# data = data.astype(np.uint16)
# print(len(data))
# print(data)
N = 64
M = 16
datatype = 'uint16'

# decimal int number to binary numpy array
def toBinary(value, wid):
    value_bin = np.binary_repr(value, width = wid)
    # width가 자릿수, value가 바꿀 정수
    
    value_bin = np.array(list(value_bin))
    value_bin = value_bin.astype(datatype)
    return value_bin

# a = 2
# print(toBinary(a, 7))

# binary numpy array to decimal int number
def toDecimal(nparr):
    fliarr = np.flip(nparr)
    # 들어온 넘피 어레이의 끝자리부터 2곱해서 더해줄려고 1차원 넘피 어레이를 뒤집어주는 함수
    # ex) [0 0 0 0 0 1] => [1 0 0 0 0 0]
    
    twoarr = np.array([2**0,2**1,2**2,2**3,2**4,2**5,2**6,2**7,2**8,2**9,2**10,2**11,2**12,2**13,2**14,2**15], dtype = np.uint16)
    valuearr = fliarr * twoarr[0:len(fliarr)]
    value = valuearr.sum()
    return value

# b = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], dtype = datatype)
# print(toDecimal(b))
# print(b.sum())

def Delta(datablock):
    baseword = datablock[0]
    delta = datablock[1:].astype(np.uint16) - baseword    
    delta = delta.astype(datatype)
    
    return baseword, delta

# baseword, delta= Delta(data)
# print(baseword)
# print(delta)
# print(len(delta))

# change to bit plane
def DBP16(deltablock):
    deltas = np.unpackbits(deltablock.view(np.uint8)).reshape(-1, 16)
    d1 = deltas[:,0:8]
    d2 = deltas[:,8:16]
    deltas = np.hstack((d2, d1))
    deltas = deltas.astype(datatype)
    
    
    dbps = np.array(list(zip(*deltas[::]))) ## 2차원 넘피 어레이를 90도 회전시키는 함수
    return dbps

# dbps16 = DBP16(delta)
# print(dbps16)
# print(dbps16.shape)

# calculate to delta-bit plane-xor
def DBX16(dbps):
    x1 = np.delete(dbps, 0, 0).astype(np.uint16)
    x2 = np.delete(dbps, dbps.shape[0] - 1, 0).astype(np.uint16)
    xored = x1 ^ x2
    xored = np.vstack([dbps[0], xored])
    return xored

# dbxs16 = DBX16(dbps16)
# print(dbxs16)
# print(dbxs16.shape)

def Encoder(dbp, dbx):
    
    global patt0
    
    # if DBX plane is all 1 or all 0
    if np.all(dbx==0):
        return 3, np.array([0, 0, 1], dtype = datatype)
    elif np.all(dbx==1):
        return 5, np.array([0, 0, 0, 0, 0], dtype = datatype)
    elif np.all(dbp==0):
        return 5, np.array([0, 0, 0, 0, 1], dtype = datatype)
    
    
    # plane is not all 1 or all 0
    pos_one = np.where(dbx == 1)[0]
    
    # single 1
    if len(pos_one) == 1:
        return 5 + 6, np.concatenate((np.array([0, 0, 0, 1, 1], dtype = datatype), toBinary(62 - pos_one[0], 6))) ## len(delta)가 이미 N-1이므로, 바로 log2를 취하면 됩니다
    
    
    elif len(pos_one) == 2:
        # Consecutive two 1 연속된 1
        if pos_one[1] - pos_one[0] == 1:
            return 5 + 6, np.concatenate((np.array([0, 0, 0, 1, 0], dtype = datatype), toBinary(62 - pos_one[1], 6))) ## 똑같이 -2가 아니라 -1만 하면 됨
        
        # 연속되지 않은 1 => not compress
        else:
            return len(delta)+1, np.concatenate((np.array([1], dtype=datatype), dbx))
        
    else:
        return len(delta)+1, np.concatenate((np.array([1], dtype=datatype), dbx))
   
# for i in range(M):
#     length, code = Encoder(dbps16[i], dbxs16[i])
#     print(dbxs16[i], "->", length, code)

def BPC(block):
    outputcode = np.array([], dtype=datatype)
    outputlen = 0
    
    baseword, deltablock = Delta(block)
    dbps = DBP16(deltablock)
    dbxs = DBX16(dbps)
    basesymbol = toBinary(baseword, 16)

    zrl = 0

    for i in range(M):
        length, code = Encoder(dbps[i], dbxs[i])

        if np.array_equal(code, np.array([0, 0, 1], dtype = datatype)): # all-0 DBX 일 경우
            zrl += 1
            continue

        else: # all-0 DBX가 아닐 경우
            if zrl != 0: # 근데 앞에 all-0 DBX가 있었을 경우
                if zrl == 1:
                    outputcode = np.concatenate((outputcode, np.array([0, 0, 1], dtype = datatype)), axis=None)
                    outputlen += 3
                else:
                    runlen = toBinary(zrl-2, 4) 
                    outputcode = np.concatenate((outputcode, np.array([0, 1], dtype=datatype), runlen), axis=None)
                    outputlen += 2 + 4
                # 이제 all-0 DBX가 아닌 현재 code를 붙여줘야함
                outputcode = np.concatenate((outputcode, code), axis=None)
                outputlen += length
                zrl = 0
            else: # 앞에 all-0 DBX가 없었을 경우
                outputcode = np.concatenate((outputcode, code), axis=None)
                outputlen += length
    
    if zrl != 0:
        if zrl == 1:
            outputcode = np.concatenate((outputcode, np.array([0, 0, 1], dtype = datatype)), axis=None)
            outputlen += 3
        else:
            runlen = toBinary(zrl-2, 4) 
            outputcode = np.concatenate((outputcode, np.array([0, 1], dtype=datatype), runlen), axis=None)
            outputlen += 2 + 4
        
    outputlen += 16
    outputcode = np.concatenate((basesymbol, outputcode), axis=None)
    
    outputlen += 2
    outputcode = np.concatenate((np.array([0, 0], dtype=datatype), outputcode), axis=None)
    
    
    # padding
    if outputlen < 512:
        e = 512 - outputlen
        ex = np.zeros(e, dtype=datatype)
        outputcode = np.concatenate((outputcode, ex), axis=None)

    
    return outputcode, outputlen

# result_bpc, len_bpc = BPC(data)
# print(result_bpc)
# print(len_bpc)
# print(len(result_bpc))

def ZRLE(block):
    outputcode = np.array([], dtype=datatype)
    outputlen = 0
    new_block = block.reshape(-1, 4)
    for idx in range(new_block.shape[0]):
        vec = np.where(new_block[idx] > 0)[0]
        if len(vec) == 0:
            outputcode = np.concatenate((outputcode, np.array([0, 0, 0, 0, 0, 0], dtype=datatype)), axis=None)
            outputlen += 6
        elif len(vec) == 1:
            if vec[0] == 3:
                outputcode = np.concatenate((outputcode, np.array([0, 0, 0, 0, 0, 1], dtype=datatype)), axis=None)
                outputlen += 6
            elif vec[0] == 2:
                outputcode = np.concatenate((outputcode, np.array([0, 0, 0, 0, 1], dtype=datatype)), axis=None)
                outputlen += 5
            elif vec[0] == 1:
                outputcode = np.concatenate((outputcode, np.array([0, 0, 0, 1, 0], dtype=datatype)), axis=None)
                outputlen += 5
            elif vec[0] == 0:
                outputcode = np.concatenate((outputcode, np.array([0, 0, 0, 1, 1], dtype=datatype)), axis=None)
                outputlen += 5
            outputcode = np.concatenate((outputcode, toBinary(new_block[idx][vec[0]], 16)), axis=None)
            outputlen += 16
        elif len(vec) == 2:
            if vec[1] == 3:
                if vec[0] == 2:
                    outputcode = np.concatenate((outputcode, np.array([0, 0, 1, 0], dtype=datatype)), axis=None)
                    outputlen += 4
                elif vec[0] == 1:
                    outputcode = np.concatenate((outputcode, np.array([0, 0, 1, 1], dtype=datatype)), axis=None)
                    outputlen += 4
                elif vec[0] == 0:
                    outputcode = np.concatenate((outputcode, np.array([0, 1, 0, 0], dtype=datatype)), axis=None)
                    outputlen += 4
            elif vec[1] == 2:
                if vec[0] == 1:
                    outputcode = np.concatenate((outputcode, np.array([0, 1, 0, 1], dtype=datatype)), axis=None)
                    outputlen += 4
                elif vec[0] == 0:
                    outputcode = np.concatenate((outputcode, np.array([0, 1, 1, 0], dtype=datatype)), axis=None)
                    outputlen += 4
            elif vec[1] == 1:
                outputcode = np.concatenate((outputcode, np.array([0, 1, 1, 1], dtype=datatype)), axis=None)
                outputlen += 4
            outputcode = np.concatenate((outputcode, toBinary(new_block[idx][vec[0]], 16), toBinary(new_block[idx][vec[1]], 16)), axis=None)
            outputlen += 32
        elif len(vec) == 3:
            sumvec = vec[0] + vec[1] + vec[2]
            if sumvec == 6: #123
                outputcode = np.concatenate((outputcode, np.array([1, 0, 0, 0], dtype=datatype)), axis=None)
                outputlen += 4
            elif sumvec == 5: #023
                outputcode = np.concatenate((outputcode, np.array([1, 0, 0, 1], dtype=datatype)), axis=None)
                outputlen += 4
            elif sumvec == 4: #013
                outputcode = np.concatenate((outputcode, np.array([1, 0, 1, 0], dtype=datatype)), axis=None)
                outputlen += 4
            elif sumvec == 3: #012
                outputcode = np.concatenate((outputcode, np.array([1, 0, 1, 1], dtype=datatype)), axis=None)
                outputlen += 4
            outputcode = np.concatenate((outputcode, toBinary(new_block[idx][vec[0]], 16), toBinary(new_block[idx][vec[1]], 16), toBinary(new_block[idx][vec[2]], 16)), axis=None)
            outputlen += 48
        elif len(vec) == 4:
            outputcode = np.concatenate((outputcode, np.array([1, 1], dtype=datatype)), axis=None)
            outputcode = np.concatenate((outputcode, toBinary(new_block[idx][vec[0]], 16), toBinary(new_block[idx][vec[1]], 16), toBinary(new_block[idx][vec[2]], 16), toBinary(new_block[idx][vec[3]], 16)), axis=None)
            outputlen += 66

    outputcode = np.concatenate((np.array([0, 1], dtype=datatype), outputcode), axis=None)
    outputlen += 2
    
    # padding
    if outputlen < 512:
        e = 512 - outputlen
        ex = np.zeros(e, dtype=datatype)
        outputcode = np.concatenate((outputcode, ex), axis=None)
    
    return outputcode, outputlen

# result_zrl, len_zrl = ZRLE(data)
# print(result_zrl)
# print(len_zrl)
# print(len(result_zrl))

def SR(block):
    reduct_block = np.logical_or(block < 128, block > 65407)
    idx_block = np.where(block > 65407)[0]
    dup_block = block.copy()
    outputcode = np.array([], dtype=datatype)
    if np.all(reduct_block == True):
        if (block[0] < 64 or block[0] > 65471):
            ret_flag = 1
        else:
            ret_flag = 0
    else:
        ret_flag = 0
    
    if (dup_block[0] > 65407):
        dup_block[0] -= 128
    dup_block = dup_block.astype(np.uint8)
    
    for idx in range(len(dup_block)):
        if idx == 0:
            outputcode = np.concatenate((outputcode, np.array([1], dtype=datatype), toBinary(dup_block[idx], 7)[-7:]), axis=None)
        else:
            outputcode = np.concatenate((outputcode, toBinary(dup_block[idx], 8)), axis=None)
    
    return outputcode, ret_flag

# result_sr, len_sr = SR(data)
# print(result_sr)
# print(len_sr)
# print(len(result_sr))

# if len_sr == 1:
#     print(result_sr)
# elif len_zrl < 513:
#     print(result_zrl)
# elif len_bpc < 513:
#     print(result_bpc)
# else:
#     print(result_bpc[:512])
import torch

def Comp(x : torch.Tensor):  
    x_numpy = x.numpy()
    newarr = np.array([], dtype=np.uint16)
    npline = np.array(list(x_numpy[:-1]), dtype=np.uint16).reshape(-1, 16)
    for i in range(64):
        newarr = np.append(newarr, toDecimal(npline[i]), axis=None)
    bpc_code, bpc_len = BPC(newarr)
    zrl_code, zrl_len = ZRLE(newarr)
    sr_code, sr_flag = SR(newarr)
    
    if sr_flag == 1:
        result = "".join(sr_code)
    elif zrl_len < 513:
        result = "".join(zrl_code)
    elif bpc_len < 513:
        result = "".join(bpc_code)
    else:
        result = "".join(bpc_code[0:512])
        