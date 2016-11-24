
from __future__ import print_function
import struct
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, Chain, Variable
from chainer.training import extensions
import argparse
import glob
import sys

class Model(Chain):
    insize=4
    def __init__(self, n_out):
        super(Model, self).__init__(
            conv1 = L.Convolution2D(None, 64, 3, pad=1), 
            conv2 = L.Convolution2D(None, 128, 3, pad=1),
            conv3 = L.Convolution2D(None, 256, 3, pad=1),
            fc4 = L.Linear(None, 4096),
            fc5 = L.Linear(None, 1024),
            fc6 = L.Linear(None, n_out),
            )
        self.train = True

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.fc4(h))
        h = F.relu(self.fc5(h))
        h = self.fc6(h)
        return h

def load_test_data(input_size, data_path):
    input_num = (input_size+1)*(input_size+1)
    forpred_data = []
    forpred_label = []
    data_list = glob.glob(data_path + "*")
    for datafileName in data_list:
        print("open ", datafileName)
        data_raw = open(datafileName)
        i = 0
        for line in data_raw:
            i = i + 1
            lineData = np.asarray([np.float32(int(x)/1023.0) for x in line.split(",")[0:input_num]])
            reshapedData = lineData.reshape(input_size+1,input_size+1)
            label = np.int32(line.split(",")[input_num])
            forpred_data.append(np.asarray([reshapedData]))
            forpred_label.append(label)
            if i > 9:
                break

    return forpred_data, forpred_label

parse = argparse.ArgumentParser(description="to use on cpp")
parse.add_argument('--filtersize','-f',type=int, default=3)
parse.add_argument('--padding','-p', type=int, default=1)
parse.add_argument('--size','-s',type=int, default=4)
parse.add_argument('--path','-pa', default="./valueDatas_of_wh4_kimono.txt")
parse.add_argument('--channels','-c', type=int, default=1)
parse.add_argument('--conv1out','-c1', type=int, default=64)
parse.add_argument('--conv2out','-c2', type=int, default=128)
parse.add_argument('--conv3out','-c3', type=int, default=256)
parse.add_argument('--fc4','-f4', type=int, default=4096)
parse.add_argument('--fc5','-f5', type=int, default=1024)
parse.add_argument('--fc6','-f6', type=int, default=35)

args = parse.parse_args()

filter_size = args.filtersize
padding = args.padding
size = args.size
channels = args.channels
c1out = args.conv1out
c2out = args.conv2out
c3out = args.conv3out
f4out = args.fc4
f5out = args.fc5
f6out = args.fc6


model = L.Classifier(Model(35))
chainer.serializers.load_npz('./my_s4_e100.model', model)

d = bytearray()

c0 = len(model.predictor.conv1.W.data)
c1 = len(model.predictor.conv1.W.data[0])
c2 = len(model.predictor.conv1.W.data[0][0])
c3 = len(model.predictor.conv1.W.data[0][0][0])
b0 = len(model.predictor.conv1.b.data)
print(c0,c1,c2,c3,b0)
for v in model.predictor.conv1.W.data.reshape(c0*c1*c2*c3):
    d += struct.pack('f',v)
for v in model.predictor.conv1.b.data:
    d += struct.pack('f',v)

    
c0 = len(model.predictor.conv2.W.data)
c1 = len(model.predictor.conv2.W.data[0])
c2 = len(model.predictor.conv2.W.data[0][0])
c3 = len(model.predictor.conv2.W.data[0][0][0])
b0 = len(model.predictor.conv2.b.data)
print(c0,c1,c2,c3,b0)
for v in model.predictor.conv2.W.data.reshape(c0*c1*c2*c3):
    d += struct.pack('f',v)
for v in model.predictor.conv2.b.data:
    d += struct.pack('f',v)
    

c0 = len(model.predictor.conv3.W.data)
c1 = len(model.predictor.conv3.W.data[0])
c2 = len(model.predictor.conv3.W.data[0][0])
c3 = len(model.predictor.conv3.W.data[0][0][0])
b0 = len(model.predictor.conv3.b.data)
print(c0,c1,c2,c3,b0)
for v in model.predictor.conv3.W.data.reshape(c0*c1*c2*c3):
    d += struct.pack('f',v)
for v in model.predictor.conv3.b.data:
    d += struct.pack('f',v)
    

l0 = len(model.predictor.fc4.W.data)
l1 = len(model.predictor.fc4.W.data[0])
b0 = len(model.predictor.fc4.b.data)
print(l0,l1,b0)
for v in model.predictor.fc4.W.data.reshape(l0*l1):
    d += struct.pack('f',v)
for v in model.predictor.fc4.b.data:
    d += struct.pack('f',v)

l0 = len(model.predictor.fc5.W.data)
l1 = len(model.predictor.fc5.W.data[0])
b0 = len(model.predictor.fc5.b.data)
print(l0,l1,b0)
for v in model.predictor.fc5.W.data.reshape(l0*l1):
    d += struct.pack('f',v)
for v in model.predictor.fc5.b.data:
    d += struct.pack('f',v)


l0 = len(model.predictor.fc6.W.data)
l1 = len(model.predictor.fc6.W.data[0])
b0 = len(model.predictor.fc6.b.data)
print(l0,l1,b0)
for v in model.predictor.fc6.W.data.reshape(l0*l1):
    d += struct.pack('f',v)
for v in model.predictor.fc6.b.data:
    d += struct.pack('f',v)
    

open("and.dat","w").write(d);

testDatas, correctLabels = load_test_data(args.size, args.path)

x = np.asarray(testDatas)
y = model.predictor(x).data
for i in np.arange(len(y)):
    print(i,x[i])
    print("  ---", np.argmax(y[i]),"---")
    print("  ",y[i])


    
