import torch 

import numpy as np

from quantize import *

def inject_no_err(input : torch.Tensor, n: int, is_ch_quant):
    with torch.no_grad():
        if is_ch_quant:
            input_q, scale, zero_p = quantize_channel(input, n)
            input[:] = dequantize_channel(input_q, scale, zero_p)
        else:
            input_q, scale, zero_p = quantize_layer(input, n)
            input[:] = dequantize_layer(input_q, scale, zero_p)            

def inject_err_default(input : torch.Tensor, p : float, n: int, is_ch_quant):
    with torch.no_grad():
        if is_ch_quant:
            input_q, scale, zero_p = quantize_channel(input, n)
        else:
            input_q, scale, zero_p = quantize_layer(input, n)
        
        input_q_1d = input_q.view((-1,))
        size = torch.numel(input_q_1d)
        err = np.packbits(np.random.binomial(1, p, size * 16))
        err_to_input = err.reshape(-1,16)[:,:8].reshape(-1)
        err_tensor = torch.tensor(err_to_input)
        input_q_1d.bitwise_xor_(err_tensor)

        if is_ch_quant:
            input[:] = dequantize_channel(input_q, scale, zero_p)
        else:
            input[:] = dequantize_layer(input_q, scale, zero_p)

def inject_err_bch(   input   : torch.Tensor, 
                    p       : float, 
                    n       : int, 
                    bch, 
                    mask,
                    is_ch_quant):
    with torch.no_grad():

        if is_ch_quant:
            input_q, scale, zero_p = quantize_channel(input, n)
        else:
            input_q, scale, zero_p = quantize_layer(input, n)
        
        size = torch.numel(input_q)
        err = np.packbits(np.random.binomial(1, p, size * 16)) & mask
        err_to_input = err.reshape(-1,16)[:,:8]
        err_to_parity = err.reshape(-1,16)[:,8:16-n] # 1, 2, 3

        err_occur = np.zeros((size // 8, 56 + n), dtype=np.uint8)
        err_occur_unpacked = np.unpackbits(err_to_input).reshape(-1, 8 * 8)
        parity = np.unpackbits(err_to_parity).reshape(-1, (8-n) * 8)

        if n == 5:
            del_col = [  0, 1, 2, 8, 9,10,16,17,18,24,25,26,
                        32,33,34,40,41,42,48,49,50,56,57,58]
            err_occur[:,:40] = np.delete(err_occur_unpacked, del_col, 1)
            err_occur[:,40:] = parity[:,:21]    
        if n == 6:
            del_col = [  0, 1, 8, 9,16,17,24,25,
                        32,33,40,41,48,49,56,57]
            err_occur[:,:48] = np.delete(err_occur_unpacked, del_col, 1)
            err_occur[:,48:] = parity[:,:14]
        if n == 7:
            del_col = [0,8,16,24,32,40,48,56]
            err_occur[:,:56] = np.delete(err_occur_unpacked, del_col, 1)
            err_occur[:,56:] = parity[:,:7]

        b = bch.decode(err_occur).reshape(-1, n)
        err_decode = np.packbits(np.fliplr(b), axis=1, bitorder='little')
        err_recovery = err_decode.reshape(-1, 8)[:,:8].reshape(-1)
        err_tensor = torch.tensor(err_recovery)

        input_q_1d = input_q.view((-1,))
        input_q_1d.bitwise_xor_(err_tensor)
        if is_ch_quant:
            input[:] = dequantize_channel(input_q, scale, zero_p)
        else:
            input[:] = dequantize_layer(input_q, scale, zero_p)

def inject_err_squid(   input   : torch.Tensor, 
                    p       : float, 
                    n       : int, 
                    encode_lut, 
                    decode_lut, 
                    rs, 
                    mask,
                    is_ch_quant):
    with torch.no_grad():
        if is_ch_quant:
            input_q, scale, zero_p = quantize_channel(input, n)
        else:
            input_q, scale, zero_p = quantize_layer(input, n)
        input_q_np = input_q.view(torch.uint8).view(-1,8).detach().numpy()
        input_q_unpacked = np.unpackbits(input_q_np).reshape(-1,8)
        input_q_weighted = input_q_unpacked * encode_lut
        input_sig = np.bitwise_xor.reduce(input_q_weighted, axis=1)
        parity = rs.encode(input_sig.reshape(-1,8))[:,8:]

        size = torch.numel(input_q)
        err = np.packbits(np.random.binomial(1, p, size * 16)) & mask
        err_to_input = err.reshape(-1,16)[:,:8]
        err_to_parity = err.reshape(-1,16)[:,8: 24 - 2 * n]
        input_w_err = input_q_np ^ err_to_input
        parity ^= (err_to_parity & 0xf)

        input_w_err_unpacked = np.unpackbits(input_w_err).reshape(-1,8)
        input_w_err_weighted = input_w_err_unpacked * encode_lut
        input_w_err_sig = np.bitwise_xor.reduce(input_w_err_weighted, axis=1)
        input_w_err_sig = input_w_err_sig.reshape(-1,8)

        read_sig = np.zeros((size // 8, 24 - 2 * n), dtype = np.uint8)   
        read_sig[:,:8] = input_w_err_sig
        read_sig[:,8:] = parity

        read_sig_decoded = rs.decode(read_sig)
        err_predicted = decode_lut[read_sig_decoded ^ input_w_err_sig]
        err_recovery = err_to_input.reshape(-1) ^ err_predicted.reshape(-1)
        err_recovery_tensor = torch.tensor(err_recovery)

        input_q_1d = input_q.view(-1)
        input_q_1d.bitwise_xor_(err_recovery_tensor)
        if is_ch_quant:
            input[:] = dequantize_channel(input_q, scale, zero_p)
        else:
            input[:] = dequantize_layer(input_q, scale, zero_p)

def inject_err_vapi(   input   : torch.Tensor, 
                    p       : float, 
                    n       : int, 
                    bch, 
                    mask,
                    is_ch_quant):
    with torch.no_grad():
        size = torch.numel(input)
        err = np.packbits(np.random.binomial(1,p, size*16))
        err_to_input = err.reshape(-1,16)[:,:8].reshape(-1)
        err_to_input_unpacked = np.unpackbits(err_to_input).reshape(-1, 8 * 8)
        err_occur = np.zeros((size // 8, 64), dtype=np.uint8)

        b = bch.decode(err_to_input_unpacked).reshape(-1,50)
        err_occur[:,::8] = b[:,:48:6]    # 10011111
        err_occur[:, 0+3: 0+3+5] = b[:, 0+1: 0+1+5]
        err_occur[:, 8+3: 8+3+5] = b[:, 6+1: 6+1+5]
        err_occur[:,16+3:16+3+5] = b[:,12+1:12+1+5]
        err_occur[:,24+3:24+3+5] = b[:,18+1:18+1+5]
        err_occur[:,32+3:32+3+5] = b[:,24+1:24+1+5]
        err_occur[:,40+3:40+3+5] = b[:,30+1:30+1+5]
        err_occur[:,48+3:48+3+5] = b[:,36+1:36+1+5]
        err_occur[:,56+1:64    ] = b[:,42+1:50    ]
        err_occur = np.packbits(err_occur)
        err_tensor = torch.tensor(err_occur).reshape(input.reshape(-1,8).shape)

        scale = 128 / (2 ** n)
        input_clip = torch.clip(input,min=-(2 **n),max=(2 ** n)).reshape(-1,8)
        sign = (input_clip < 0).int() ^ ((err_tensor & 0x80) >> 7)
        value = torch.abs(input_clip).reshape(-1,8)
        value_temp = torch.round(value * scale).int() ^ (err_tensor & 0x7f)
        det  = (value_temp < 32 ) * 128 + (value_temp - 32)
        idx  = torch.argmin(det, dim=1)

        mask = torch.eye(8,dtype=torch.uint8)[idx] | (value < 0.5)
        value_temp = (value_temp & 0xfc) + (value_temp & 0x3) * mask
        input[:] = ((sign.reshape(input.shape) * (-2) + 1) * value_temp.reshape(input.shape) / scale)


def inject_multi_err_vapi(   input   : torch.Tensor, 
                    p       : float, 
                    n       : int, 
                    bch, 
                    mask,
                    is_ch_quant):
    with torch.no_grad():
        size = torch.numel(input)
        err_origin = np.random.binomial(1, p, size * 16)
        err_idx = np.nonzero(err_origin)[0]

        err_unpacked = np.zeros(size * 16 + 8, dtype=np.uint8)

        err_unpacked[err_idx] = 1
        err_unpacked[err_idx + 1] = 1

        err = np.packbits(err_unpacked[:size * 16])
        err_to_input = err.reshape(-1,16)[:,:8].reshape(-1)
        err_to_input_unpacked = np.unpackbits(err_to_input).reshape(-1, 8 * 8)
        err_occur = np.zeros((size // 8, 64), dtype=np.uint8)

        b = bch.decode(err_to_input_unpacked).reshape(-1,50)
        err_occur[:,::8] = b[:,:48:6]    # 10011111
        err_occur[:, 0+3: 0+3+5] = b[:, 0+1: 0+1+5]
        err_occur[:, 8+3: 8+3+5] = b[:, 6+1: 6+1+5]
        err_occur[:,16+3:16+3+5] = b[:,12+1:12+1+5]
        err_occur[:,24+3:24+3+5] = b[:,18+1:18+1+5]
        err_occur[:,32+3:32+3+5] = b[:,24+1:24+1+5]
        err_occur[:,40+3:40+3+5] = b[:,30+1:30+1+5]
        err_occur[:,48+3:48+3+5] = b[:,36+1:36+1+5]
        err_occur[:,56+1:64    ] = b[:,42+1:50    ]
        err_occur = np.packbits(err_occur)
        err_tensor = torch.tensor(err_occur).reshape(input.reshape(-1,8).shape)

        scale = 128 / (2 ** n)
        input_clip = torch.clip(input,min=-(2 **n),max=(2 ** n)).reshape(-1,8)
        sign = (input_clip < 0).int() ^ ((err_tensor & 0x80) >> 7)
        value = torch.abs(input_clip).reshape(-1,8)
        value_temp = torch.round(value * scale).int() ^ (err_tensor & 0x7f)
        det  = (value_temp < 32 ) * 128 + (value_temp - 32)
        idx  = torch.argmin(det, dim=1)

        mask = torch.eye(8,dtype=torch.uint8)[idx] | (value < 0.5)
        value_temp = (value_temp & 0xfc) + (value_temp & 0x3) * mask
        input[:] = ((sign.reshape(input.shape) * (-2) + 1) * value_temp.reshape(input.shape) / scale)


def inject_multi_err_squid(input   : torch.Tensor, 
                        p       : float, 
                        n       : int, 
                        encode_lut, 
                        decode_lut, 
                        rs, 
                        mask,
                        is_ch_quant):
    with torch.no_grad():
        if is_ch_quant:
            input_q, scale, zero_p = quantize_channel(input, n)
        else:
            input_q, scale, zero_p = quantize_layer(input, n)
        input_q_np = input_q.view(torch.uint8).view(-1,8).detach().numpy()
        input_q_unpacked = np.unpackbits(input_q_np).reshape(-1,8)
        input_q_weighted = input_q_unpacked * encode_lut
        input_sig = np.bitwise_xor.reduce(input_q_weighted, axis=1)
        parity = rs.encode(input_sig.reshape(-1,8))[:,8:]

        size = torch.numel(input_q)
        err_origin = np.random.binomial(1, p, size * 16)
        err_idx = np.nonzero(err_origin)[0]

        err_unpacked = np.zeros(size * 16 + 8, dtype=np.uint8)

        err_unpacked[err_idx] = 1
        err_unpacked[err_idx + 1] = 1

        err = np.packbits(err_unpacked[:size * 16]) & mask
        err_to_input = err.reshape(-1,16)[:,:8]
        err_to_parity = err.reshape(-1,16)[:,8: 24 - 2 * n]
        input_w_err = input_q_np ^ err_to_input
        parity ^= (err_to_parity & 0xf)

        input_w_err_unpacked = np.unpackbits(input_w_err).reshape(-1,8)
        input_w_err_weighted = input_w_err_unpacked * encode_lut
        input_w_err_sig = np.bitwise_xor.reduce(input_w_err_weighted, axis=1)
        input_w_err_sig = input_w_err_sig.reshape(-1,8)

        read_sig = np.zeros((size // 8, 24 - 2 * n), dtype = np.uint8)   
        read_sig[:,:8] = input_w_err_sig
        read_sig[:,8:] = parity

        read_sig_decoded = rs.decode(read_sig)
        err_predicted = decode_lut[read_sig_decoded ^ input_w_err_sig]
        err_recovery = err_to_input.reshape(-1) ^ err_predicted.reshape(-1)
        err_recovery_tensor = torch.tensor(err_recovery)

        input_q_1d = input_q.view(-1)
        input_q_1d.bitwise_xor_(err_recovery_tensor)
        if is_ch_quant:
            input[:] = dequantize_channel(input_q, scale, zero_p)
        else:
            input[:] = dequantize_layer(input_q, scale, zero_p)

def inject_multi_err_bch(   input   : torch.Tensor, 
                    p       : float, 
                    n       : int, 
                    bch, 
                    mask,
                    is_ch_quant):
    with torch.no_grad():

        if is_ch_quant:
            input_q, scale, zero_p = quantize_channel(input, n)
        else:
            input_q, scale, zero_p = quantize_layer(input, n)
        
        size = torch.numel(input_q)


        size = torch.numel(input_q)
        err_origin = np.random.binomial(1, p, size * 16)
        err_idx = np.nonzero(err_origin)[0]

        err_unpacked = np.zeros(size * 16 + 8, dtype=np.uint8)

        err_unpacked[err_idx] = 1
        err_unpacked[err_idx + 1] = 1

        err = np.packbits(err_unpacked[:size * 16]) & mask
        err_to_input = err.reshape(-1,16)[:,:8]
        err_to_parity = err.reshape(-1,16)[:,8:16-n] # 1, 2, 3

        err_occur = np.zeros((size // 8, 56 + n), dtype=np.uint8)
        err_occur_unpacked = np.unpackbits(err_to_input).reshape(-1, 8 * 8)
        parity = np.unpackbits(err_to_parity).reshape(-1, (8-n) * 8)

        if n == 5:
            del_col = [  0, 1, 2, 8, 9,10,16,17,18,24,25,26,
                        32,33,34,40,41,42,48,49,50,56,57,58]
            err_occur[:,:40] = np.delete(err_occur_unpacked, del_col, 1)
            err_occur[:,40:] = parity[:,:21]    
        if n == 6:
            del_col = [  0, 1, 8, 9,16,17,24,25,
                        32,33,40,41,48,49,56,57]
            err_occur[:,:48] = np.delete(err_occur_unpacked, del_col, 1)
            err_occur[:,48:] = parity[:,:14]
        if n == 7:
            del_col = [0,8,16,24,32,40,48,56]
            err_occur[:,:56] = np.delete(err_occur_unpacked, del_col, 1)
            err_occur[:,56:] = parity[:,:7]

        b = bch.decode(err_occur).reshape(-1, n)
        err_decode = np.packbits(np.fliplr(b), axis=1, bitorder='little')
        err_recovery = err_decode.reshape(-1, 8)[:,:8].reshape(-1)
        err_tensor = torch.tensor(err_recovery)

        input_q_1d = input_q.view((-1,))
        input_q_1d.bitwise_xor_(err_tensor)
        if is_ch_quant:
            input[:] = dequantize_channel(input_q, scale, zero_p)
        else:
            input[:] = dequantize_layer(input_q, scale, zero_p)

def inject_err_weight_null( input : torch.Tensor, 
                            p: float, nbit: int, mask):
    with torch.no_grad():
        input_1d = input.view(torch.int16).view((-1,))
        input_1d &= 0xfffe
        size = torch.numel(input_1d)

        err_origin = np.random.binomial(1, p, size * 16)
        err_idx = np.nonzero(err_origin)[0]
        err_unpacked = np.zeros(size * 16 + 8, dtype=np.uint8)
        err_unpacked[err_idx]   = 1
        err_unpacked[err_idx+1] = 1
        err = np.packbits(err_unpacked[:size * 16]).view(np.int16)
        
        flag = (err & 0x5555) + ((err >> 1) & 0x5555)
        flag = (flag & 0x3333) + ((flag >> 2) & 0x3333)
        flag = (flag & 0x0f0f) + ((flag >> 4) & 0x0f0f)
        flag = (flag & 0x00ff) + ((flag >> 8) & 0x00ff)
        weight_null = torch.tensor(flag & 1) ^ 1
        err = torch.tensor(err)
        input_1d.bitwise_xor_(err)
        input_1d.mul_(weight_null)
