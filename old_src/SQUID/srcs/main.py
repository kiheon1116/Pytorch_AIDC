import os

from setting import *
from model import Model, ErrorModel

def get_name(args):
    name = f'{args.model}_int{args.nbit}({args.quant})_{args.option}'
    name += f'_{-args.ber:.1f}_{args.g_seed:d}_{args.iteration_start}'   
    return name 

def get_name_bitpos(args):
    name = f'{args.model}_int{args.nbit}({args.quant})_{args.option}_{args.bit_pos}'
    name += f'_{-args.ber:.1f}_{args.g_seed:d}_{args.iteration_start}'   
    return name 

def check_and_make_dir(args, string):
    path = f'{args.save_path}/{string}/{args.option}/{args.model}'
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

def get_ber_list(args):
    ber_start = args.ber    
    ber_stride = args.ber_stride 
    ber_step = 1 / ber_stride
    ber_list = [ber_start + num * ber_step for num in range(ber_stride)]
    return ber_list

def main(args):
    set_environment(args)
    if args.option == 'bit_pos':
        name = get_name_bitpos(args)
    else:
        name = get_name(args)
    log_save_path = check_and_make_dir(args, 'log')
    log_file_name = f'{log_save_path}/{name}_ber{args.ber}.log'

    total = 0
    with open(log_file_name, "w", 1) as log_file:
        for it in range(args.iteration):
            args.l_seed = manual_l_seed()
            model = ErrorModel(args)
            acc1 = model.validate()
            total += acc1
            print(f'{">" * 8} Acc : {acc1:.3f}%')
            log_file.write(f'{it:02d}|seed{args.l_seed:010d}|,')
            log_file.write(f'{args.ber:.1f},{acc1:.3f}\n')
    return total / args.iteration

if __name__ == '__main__':
    args = parse_argument()
    if args.option == 'bit_pos':
        name = get_name_bitpos(args)
    else:
        name = get_name(args)
    result_save_path = check_and_make_dir(args, 'csv')
    result_file_name = f'{result_save_path}/{name}.csv'

    with open(result_file_name, "w", 1) as result_file:
        ber_list = get_ber_list(args)
        for ber in ber_list:
            args.ber = ber
            result = main(args)
            result_file.write(f'{ber:.1f},{result:.3f}\n')

    