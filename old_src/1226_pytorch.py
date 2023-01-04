import numpy as np
import matplotlib.pyplot as plt
import argparse

def parse_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument('--filename', default = '/home/kkh/')
    return parser.parse_args()


def main(args):
    #print(args.filename)
    index = np.load(args.filename).reshape(-1)
    plt.xlabel("real value bit position (n ~ n+7)\n(2^n ~ 2^(n+7))\n")
    plt.ylabel("counts")
    plt.xticks(range(1,12))
    plt.hist(index, bins = 11, range=(1,12) ,histtype='bar',align='left', rwidth=0.8)
    #plt.show()
    plt.savefig(args.filename+".png",dpi=500)



if __name__ == '__main__':
    args = parse_argument()
    main(args)
