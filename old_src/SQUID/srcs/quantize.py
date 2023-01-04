import torch

@torch.jit.script
def quantize_layer(x: torch.Tensor, n : int):

    N = 2 ** n
    N_MIN, N_MAX = -N//2, N//2 - 1
    x_max, x_min = torch.max(x) , torch.min(x)

    scale = (x_max - x_min) / (N-1)
    scale += (x_max * (scale == 0))
    zero_n = x_max * N_MIN - x_min * N_MAX
    zero_d = x_max - x_min
    zero_p =  torch.round(zero_n / (zero_d + 1e-30)) * (zero_d != 0)

    x_hat = torch.round(x / scale + zero_p)
    x_q   = torch.clip(x_hat, N_MIN, N_MAX).type(torch.int8)

    return x_q, scale, zero_p

@torch.jit.script
def dequantize_layer(   x_q: torch.Tensor, 
                        scale: torch.Tensor, 
                        zero_p: torch.Tensor):
    return scale  * (x_q - zero_p)

@torch.jit.script
def quantize_channel(x: torch.Tensor, n : int):

    N = 2 ** n
    N_MIN, N_MAX = -N//2, N//2 - 1
    
    if len(x.shape) >= 4:
        x_2d  = x.view(x.shape[0], -1)
        x_max = torch.max(x_2d,dim=1)[0]
        x_min = torch.min(x_2d,dim=1)[0]

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

    x_max, x_min = torch.max(x) , torch.min(x)

    scale = (x_max - x_min) / (N-1)
    scale += (x_max * (scale == 0))
    zero_n = x_max * N_MIN - x_min * N_MAX
    zero_d = x_max - x_min
    zero_p = torch.round(zero_n / (zero_d + 1e-30)) * (zero_d != 0)

    x_hat = torch.round(x / scale + zero_p)
    x_q   = torch.clip(x_hat, N_MIN, N_MAX).type(torch.int8)

    return x_q, scale, zero_p

@torch.jit.script
def dequantize_channel  (   x_q: torch.Tensor, 
                            scale: torch.Tensor, 
                            zero_p: torch.Tensor):

    x_2d = x_q.view(x_q.shape[0], -1)
    return (scale  * (x_2d - zero_p)).view(x_q.shape)

@torch.jit.script
def quantize_fixed(x : torch.Tensor, n : int):
    result = torch.zeros(size = x.shape)
    sign = (x < 0) * (-1) + (x > 0) * 1

    m = 1.0
    y = torch.abs(x)
    for _ in range(7):
        result += (y >= m) * 64 * m
        y -= ((y >= m) * m)
        m /= 2
    result /= 64 
    up =  torch.abs((result + 1/64) - torch.abs(x))
    down = torch.abs(result - torch.abs(x))

    result = (up >= down) * result + (up < down) * (result + 1/64)
    result *= sign
    return result
