from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SigmoidLayer,LinearLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised import BackpropTrainer
from pickle import dump,load
from random import shuffle
from copy import deepcopy
from visual.graph import *

data_path = '/home/tzoorp/Desktop/yoav/data/converted_data.txt'
base = '/home/tzoorp/Desktop/yoav/networks/'
AVG_TO_CRT = [[-2],[-1]]
TESTS_TO_CRT = [range(11),[-1]]
TESTS_TO_AVG = [range(11),[-2]]

def buildDataset(path,indexes):
    f = open(path)
    ds = SupervisedDataSet(len(indexes[0]),len(indexes[1]))
    indexin,indexout = indexes
    for line in f.readlines():
        outline = [float(x) for x in line.split('\t')[:-1]]
        inpt,outpt = [],[]
        for i in indexin:
            inpt.append(outline[i])
        for i in indexout:
            outpt.append(outline[i])
        ds.appendLinked(inpt,outpt)
    return ds

def splitDataset(ds,ratio=2.0/3):
    indim,outdim = len(ds['input'][0]),len(ds['target'][0])
    dsT = SupervisedDataSet(indim,outdim)
    dsV = SupervisedDataSet(indim,outdim)

    data = zip(ds['input'],ds['target'])
    shuffle(data)
    choice = [dsT]*int(len(data)*ratio+1)+[dsV]*int(len(data)*(1-ratio))
    for i in range(len(data)):
        choice[i].appendLinked(*data[i])
    return (dsT,dsV)
    
def createTrainer(dsPath,indexes,sizes,hidden_class=SigmoidLayer,out_class=LinearLayer):
    ds = buildDataset(dsPath,indexes)
    net = buildNetwork(*sizes,hiddenclass=hidden_class,outclass=out_class,bias=False)
    dsT,dsV = splitDataset(ds)
    return BackpropTrainer(net,dsT),dsV

def saveTrainer(trainer,dsV,path_trainer,path_dsV):
    dump(trainer,open(path_trainer,'w'))
    dump(dsV,open(path_dsV,'w'))

def loadTrainer(path_trainer,path_dsV):
    return load(open(path_trainer)),load(open(path_dsV))

def trainMinError(trainer,dsV,minTrainer=None,batch_size=0,epochs=50,plotErr=False,i0=0):
    dsT = trainer.ds
    if minTrainer == None:
       minTrainer = deepcopy(trainer)
    for i in range(epochs):
        if batch_size==0:
            ds = dsT
        else:
            ds = SupervisedDataSet(len(dsT['input'][0]),len(dsT['target'][0]))
            data = zip(dsT['input'],dsT['target'])
            shuffle(data)
            for k in range(batch_size):
                ds.appendLinked(data[k][0],data[k][1])
        trainer.ds = ds
        trainer.train()
        TE = trainer.testOnData(dsT)
        VE = trainer.testOnData(dsV)
        MVE = minTrainer.testOnData(dsV)
        if VE<MVE:
            minTrainer = BackpropTrainer(deepcopy(trainer.module),dsT)
        if plotErr:
            plotError(i+i0,TE,VE,MVE)
    trainer.ds = dsT
    return minTrainer

def printNetwork(net):
    for mod in net.module.modules:
        for conn in net.module.connections[mod]:
            print conn
            for cc in range(len(conn.params)):
                print conn.whichBuffers(cc), conn.params[cc]

def plotError(n,trainError,verError,minError):
    init_graphics()
    global train_curve,ver_curve,min_curve
    train_curve.plot(pos=[n,trainError])
    ver_curve.plot(pos=[n,verError])
    min_curve.plot(pos=[n,minError])
    
def init_graphics():
    global gd,train_curve,ver_curve,min_curve
    try:
        gd
    except NameError:
        gd = gdisplay()
    try:
        train_curve
    except NameError:
        train_curve = gcurve(color=color.red)
    try:
        ver_curve
    except NameError:
        ver_curve = gcurve(color=color.blue)
    try:
        min_curve
    except NameError:
        min_curve = gcurve(color=color.green)
    

def main_tests_to_avg():
    trainer,dsV = createTrainer(data_path,TESTS_TO_AVG,[11,3,1],LinearLayer,LinearLayer)
    minTrainer = trainMinError(trainer,dsV,None,500,100,True)
    print "Trainer: ",trainer.testOnData(dsV)
    print "Min Trainer: ",minTrainer.testOnData(dsV)
    printNetwork(minTrainer)
