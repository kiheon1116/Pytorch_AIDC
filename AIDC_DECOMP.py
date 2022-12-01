import numpy as np
import math
import time
import sys

def sr_decomp(data_i):
    data_o = np.array((), dtype="int")
    if data_i[1] == 1:
        code0 = np.array([1,1,1,1,1,1,1,1,1], dtype="int")
    else:
        code0 = np.array([0,0,0,0,0,0,0,0,0], dtype="int")
    code0 = np.concatenate((code0, data_i[1:8]), dtype="int")
    data_o = np.concatenate((data_o, code0), dtype="int", axis=0)
    for i in range(1,64):
        if data_i[i*8] == 1:
            code_i = np.array([1,1,1,1,1,1,1,1], dtype="int")
        else:
            code_i = np.array([0,0,0,0,0,0,0,0], dtype="int")
        code_i = np.concatenate((code_i, data_i[i*8:(i+1)*8]), dtype="int")
        data_o = np.concatenate((data_o, code_i), dtype="int")
    return data_o

def zrl_decomp(data_i):
    data_o = np.array((), dtype="int")
    cnt = 0
    idx = 2
    while cnt < 16:
        if np.array_equal(data_i[idx:idx+2], np.array([1,1], dtype="int")):
            code_i=data_i[idx+2:idx+66]
            cnt=cnt+1
            idx=idx+66
        elif np.array_equal(data_i[idx:idx+4], np.array([1,0,1,1], dtype="int")):
            code_i=np.zeros(64, dtype="int")
            code_i[0:16]=data_i[idx+4:idx+20]
            code_i[16:32]=data_i[idx+20:idx+36]
            code_i[32:48]=data_i[idx+36:idx+52]
            cnt=cnt+1
            idx=idx+52
        elif np.array_equal(data_i[idx:idx+4], np.array([1,0,1,0], dtype="int")):
            code_i=np.zeros(64, dtype="int")
            code_i[0:16]=data_i[idx+4:idx+20]
            code_i[16:32]=data_i[idx+20:idx+36]
            code_i[48:64]=data_i[idx+36:idx+52]
            cnt=cnt+1
            idx=idx+52
        elif np.array_equal(data_i[idx:idx+4], np.array([1,0,0,1], dtype="int")):
            code_i=np.zeros(64, dtype="int")
            code_i[0:16]=data_i[idx+4:idx+20]
            code_i[32:48]=data_i[idx+20:idx+36]
            code_i[48:64]=data_i[idx+36:idx+52]
            cnt=cnt+1
            idx=idx+52
        elif np.array_equal(data_i[idx:idx+4], np.array([1,0,0,0], dtype="int")):
            code_i=np.zeros(64, dtype="int")
            code_i[16:32]=data_i[idx+4:idx+20]
            code_i[32:48]=data_i[idx+20:idx+36]
            code_i[48:64]=data_i[idx+36:idx+52]
            cnt=cnt+1
            idx=idx+52
        elif np.array_equal(data_i[idx:idx+4], np.array([0,1,1,1], dtype="int")):
            code_i=np.zeros(64, dtype="int")
            code_i[0:16]=data_i[idx+4:idx+20]
            code_i[16:32]=data_i[idx+20:idx+36]
            cnt=cnt+1
            idx=idx+36
        elif np.array_equal(data_i[idx:idx+4], np.array([0,1,1,0], dtype="int")):
            code_i=np.zeros(64, dtype="int")
            code_i[0:16]=data_i[idx+4:idx+20]
            code_i[32:48]=data_i[idx+20:idx+36]
            cnt=cnt+1
            idx=idx+36
        elif np.array_equal(data_i[idx:idx+4], np.array([0,1,0,1], dtype="int")):
            code_i=np.zeros(64, dtype="int")
            code_i[16:32]=data_i[idx+4:idx+20]
            code_i[32:48]=data_i[idx+20:idx+36]
            cnt=cnt+1
            idx=idx+36
        elif np.array_equal(data_i[idx:idx+4], np.array([0,1,0,0], dtype="int")):
            code_i=np.zeros(64, dtype="int")
            code_i[0:16]=data_i[idx+4:idx+20]
            code_i[48:64]=data_i[idx+20:idx+36]
            cnt=cnt+1
            idx=idx+36
        elif np.array_equal(data_i[idx:idx+4], np.array([0,0,1,1], dtype="int")):
            code_i=np.zeros(64, dtype="int")
            code_i[16:32]=data_i[idx+4:idx+20]
            code_i[48:64]=data_i[idx+20:idx+36]
            cnt=cnt+1
            idx=idx+36
        elif np.array_equal(data_i[idx:idx+4], np.array([0,0,1,0], dtype="int")):
            code_i=np.zeros(64, dtype="int")
            code_i[32:48]=data_i[idx+4:idx+20]
            code_i[48:64]=data_i[idx+20:idx+36]
            cnt=cnt+1
            idx=idx+36
        elif np.array_equal(data_i[idx:idx+5], np.array([0,0,0,1,1], dtype="int")):
            code_i=np.zeros(64, dtype="int")
            code_i[0:16]=data_i[idx+5:idx+21]
            cnt=cnt+1
            idx=idx+21
        elif np.array_equal(data_i[idx:idx+5], np.array([0,0,0,1,0], dtype="int")):
            code_i=np.zeros(64, dtype="int")
            code_i[16:32]=data_i[idx+5:idx+21]
            cnt=cnt+1
            idx=idx+21
        elif np.array_equal(data_i[idx:idx+5], np.array([0,0,0,0,1], dtype="int")):
            code_i=np.zeros(64, dtype="int")
            code_i[32:48]=data_i[idx+5:idx+21]
            cnt=cnt+1
            idx=idx+21
        elif np.array_equal(data_i[idx:idx+6], np.array([0,0,0,0,0,1], dtype="int")):
            code_i=np.zeros(64, dtype="int")
            code_i[48:64]=data_i[idx+6:idx+22]
            cnt=cnt+1
            idx=idx+22
        elif np.array_equal(data_i[idx:idx+6], np.array([0,0,0,0,0,0], dtype="int")):
            code_i=np.zeros(64, dtype="int")
            cnt=cnt+1
            idx=idx+6
        data_o=np.concatenate((data_o, code_i), dtype="int")

    return data_o

def bpc_decomp(data_i):
    data_o = np.array((), dtype="int")
    idx = 18;
    base = data_i[2:18]
    dbps = Decoder(data_i[18:512]) # make dbps with 494bit data
    
    deltas = Bitplane(dbps)
    
    origin = Adder(base, deltas)
    
    for i in range(16):
        code_o = origin[i]
        data_o = np.concatenate((data_o,code_o), dtype="int")
    return data_o

def toDecimal(nparr):
    fliarr = np.flip(nparr)
    twoarr = np.array([2**0,2**1,2**2,2**3,2**4,2**5,2**6,2**7,2**8,2**9,2**10,2**11,2**12,2**13,2**14,2**15], dtype="int")
    valuearr = fliarr * twoarr[0:len(fliarr)]
    value = valuearr.sum()
    return value

def toBinary(value, wid):
    value_bin = np.binary_repr(value, width=wid)
    if len(value_bin) > wid:
        sub = len(value_bin) - wid
        value_bin = value_bin[sub:sub+16]
    value_bin = np.array(list(value_bin), dtype="int")
    return value_bin

def Decoder(data_i):
    dbps_d = np.zeros((16,63), dtype="int")
    idx_d = 0
    zrl_cnt = 0
    do_xor = 0
    cnt_d = 0
    
    while (cnt_d < 16) & (idx_d < 494):
        if zrl_cnt > 0:
            code = np.zeros(63, dtype="int")
            do_xor=1
            zrl_cnt = zrl_cnt-1;
        else :
            if data_i[idx_d]:
                if idx_d < 430:
                    code=data_i[idx_d+1:idx_d+64]
                    do_xor=1
                    idx_d=idx_d+64
                else :
                    code=np.zeros(63, dtype="int")
                    do_xor=0
                    idx_d=494
            elif np.array_equal(data_i[idx_d:idx_d+2], np.array([0,1], dtype="int")):
                if idx_d < 488:
                    code = np.zeros(63, dtype="int")
                    do_xor=1
                    zrl_cnt = toDecimal(data_i[idx_d+2:idx_d+6])+1
                    idx_d=idx_d+6
                else :
                    code = np.zeros(63, dtype="int")
                    do_xor=0
                    idx_d=494
            elif np.array_equal(data_i[idx_d:idx_d+3], np.array([0,0,1], dtype="int")):
                if idx_d < 489:
                    code = np.zeros(63, dtype="int")
                    do_xor=1
                    idx_d=idx_d+3
                else :
                    code = np.zeros(63, dtype="int")
                    do_xor=0
                    idx_d=494
            elif np.array_equal(data_i[idx_d:idx_d+5], np.array([0,0,0,0,0], dtype="int")):
                if idx_d < 489:
                    code = np.ones(63, dtype="int")
                    do_xor=1
                    idx_d=idx_d+5
                else :
                    code = np.zeros(63, dtype="int")
                    do_xor=0
                    idx_d=494
            elif np.array_equal(data_i[idx_d:idx_d+5], np.array([0,0,0,0,1], dtype="int")): # do not xor
                if idx_d < 489:
                    code = np.zeros(63, dtype="int")
                    do_xor=0
                    idx_d=idx_d+5
                else :
                    code = np.zeros(63, dtype="int")
                    do_xor=0
                    idx_d=494
            elif np.array_equal(data_i[idx_d:idx_d+5], np.array([0,0,0,1,0], dtype="int")):
                if idx_d < 483:
                    code = np.zeros(63, dtype="int")
                    bin_pos=data_i[idx_d+5:idx_d+11]
                    pos=toDecimal(bin_pos)
                    code[pos]=1
                    code[pos+1]=1
                    do_xor=1
                    idx_d=idx_d+11
                else :
                    code = np.zeros(63, dtype="int")
                    do_xor=0
                    idx_d=494
            elif np.array_equal(data_i[idx_d:idx_d+5], np.array([0,0,0,1,1], dtype="int")):
                if idx_d < 483:
                    code = np.zeros(63, dtype="int")
                    bin_pos=data_i[idx_d+5:idx_d+11]
                    pos=toDecimal(bin_pos)
                    code[pos]=1
                    do_xor=1
                    idx_d=idx_d+11
                else :
                    code = np.zeros(63, dtype="int")
                    do_xor=0
                    idx_d=494
        
        if cnt_d == 0:
            dbps_d[cnt_d] = code
        else:
            if do_xor:
                dbps_d[cnt_d] = code ^ dbps_d[cnt_d-1]
            else:
                dbps_d[cnt_d] = code
        do_xor=0
        
        cnt_d=cnt_d+1
    return dbps_d

def Bitplane(dbps):
    deltas = np.array(list(zip(*dbps[::])))
    return deltas

def adderBintoBin(num1, num2):
    deci_num1 = toDecimal(num1)
    deci_num2 = toDecimal(num2)
    out_num = deci_num1 + deci_num2
    bin_out = toBinary(out_num, 16)
    return bin_out

def Adder(base, deltas):
    origin = np.zeros((16,64), dtype="int")
    idx=0
    while idx < 16:
        if idx == 0:
            data_1000 = base
        else :
            data_1000 = adderBintoBin(base, deltas[4*idx-1])
        data_0100 = adderBintoBin(base, deltas[4*idx])
        data_0010 = adderBintoBin(base, deltas[4*idx+1])
        data_0001 = adderBintoBin(base, deltas[4*idx+2])
        
        origin[idx] = np.concatenate((data_1000, data_0100, data_0010, data_0001), dtype="int")
        idx=idx+1
    return origin

def DECOMP_TOP(data_i):
    if data_i[0:2] == '00': mode = 0
    elif data_i[0:2] == '01': mode = 1
    elif data_i[0] == '1': mode = 2
    else : mode = 3 # error
        
    ndata_i = np.array(list(data_i[:-1]), dtype="int") # without \n
    
    if mode == 0:
        data_o = bpc_decomp(ndata_i)
    elif mode == 1:
        data_o = zrl_decomp(ndata_i)
    elif mode == 2:
        data_o = sr_decomp(ndata_i)
    else :
        data_o = np.array([], dtype="int")
    str_data_o = "".join(data_o.astype('str'))
    return str_data_o, mode

def DECOMP():
     read_f = open("/home/us03145/python/weight_out_1.txt", 'r')
     write_f = open("/home/us03145/python/FCLweight_from_decomp.txt", 'w')
     compr_f = open("/home/us03145/python/FCL_weight.txt", 'r')

     bcnt=1
     su_cnt=0
     fl_cnt=0
     zrlcnt=0
     bpccnt=0
     srcnt=0

     while True:
          data_i = read_f.readline()
          cdata_i = compr_f.readline()
          if data_i == "": break;
               
          ndata_i = data_i
          ncdata_i = cdata_i[:-1]
          
          data_o, mode = DECOMP_TOP(data_i)

          if ncdata_i == data_o : 
               su_cnt=su_cnt+1
          else :
               fl_cnt=fl_cnt+1
               
          if mode == 0:
               bpccnt=bpccnt+1
          elif mode == 1:
               zrlcnt=zrlcnt+1
          elif mode == 2:
               srcnt=srcnt+1
               
          bcnt=bcnt+1
          write_f.write(data_o)
          write_f.write("\n")

     print(f'compression with bpc = {bpccnt}')
     print(f'compression with zrl = {zrlcnt}')
     print(f'compression with sr  = {srcnt}')
     print(f'success count = {su_cnt}')
     print(f'success rate = {su_cnt/bcnt*100}%')
     print(f'faliure count = {fl_cnt}')
     print(f'faliure rate = {fl_cnt/bcnt*100}%')
     read_f.close()
     write_f.close()
     compr_f.close()