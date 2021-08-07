import numpy as np
import os
import qrcode
import argparse

chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
         'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
         'w', 'x', 'y', 'z',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
         'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z',
         '!', '@', '#', '$', '%', '^', '&', '*', '(', ')',
         '-', '+', '/', '?', ',']

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--number', '-n', type=int,
                        default=1000, help='How many qrcode images will be generated')
    parser.add_argument('--min_length', '-min', type=int,
                        default=10, help='min length of the string encoded in qrcode')
    parser.add_argument('--max_length', '-max', type=int,
                        default=25, help='max length of the string encoded in qrcode')
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                        help='Dir to save the result')
    parser.add_argument('--size', '-s', type=int, default=6,
                        help='qrcode image pixel size')
    parser.add_argument('--version', '-v', type=int, default=1,
                        help='version of the qrcode')
    args = parser.parse_args()
    return args

def random_length(min_length, max_length):
    return np.random.randint(min_length, max_length)

def random_index(length):
    return np.random.randint(0, len(chars), size=(length,))

def string_from_index(index):
    s = ''
    for i in index:
        s += chars[i]
    return s

def string2qrcode(string, version, size):
    img = qrcode.make(string, version=version, box_size=size)
    return img

def run(args):
    for i in range(args.number):
        print("Generating {}/{} qrcode image".format(i, args.number))
        length = random_length(args.min_length, args.max_length)
        index = random_index(length)
        string = string_from_index(index)
        img = string2qrcode(string, args.version, args.size)
        imgname = '{:04d}.jpg'.format(i)
        imgname = os.path.join(args.output_dir, imgname)
        img.save(imgname)

def main():
    args = get_args()
    run(args)

if __name__ == '__main__':
    main()