import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--src', default='data/test.txt', help='source tag file')
parser.add_argument('--pred', default='test.98', help='prediction')
parser.add_argument('--dst', default='test.eval', help='result store')
args = parser.parse_args()

with open(args.src, 'r') as f:
    data = []
    for line in f:
        if len(line) <= 1: data.append('\n')
        else: data.append(' '.join(line[:-1].split()))
with open(args.pred, 'r') as f:
    data1 = []
    for line in f:
        if len(line) <= 1: data1.append('\n')
        else: data1.append(line[:-1])
with open(args.dst, 'w') as f:
    for d, d1 in zip(data, data1):
        if d == '\n': f.write('\n')
        else: f.write('{} {}\n'.format(d, d1))
