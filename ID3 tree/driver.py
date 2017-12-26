import ID3Algo
import sys
sys.setrecursionlimit(10000)
import argparse
import csv
import pickle as pkl
# You can test the algorithm by use simple_feature and simple label
# instead of trainf and testf

def load_data(ftrain, ftest):
    Xtrain, Ytrain, Xtest, Ytest = [],[],[],[]
    with open(ftrain, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            rw = map(int, row[0].split())
            Xtrain.append(rw)

    with open(ftest, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            rw = map(int, row[0].split())
            Xtest.append(rw)

    ftrain_label = ftrain.split('.')[0] + '_label.csv'
    with open(ftrain_label, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            rw = int(row[0])
            Ytrain.append(rw)

    print('Data Loading: done')
    return Xtrain, Ytrain, Xtest



parser = argparse.ArgumentParser()
parser.add_argument('-p', required=True)
parser.add_argument('-f1', help='training file in csv format', required=True)
parser.add_argument('-f2', help='test file in csv format', required=True)
parser.add_argument('-o', help='output labels for the test dataset', required=True)
parser.add_argument('-t', help='output tree filename', required=True)

args = vars(parser.parse_args())

pval = args['p']
Xtrain_name = args['f1']
Ytrain_name = args['f1'].split('.')[0]+ '_labels.csv' #labels filename will be the same as training file name but with _label at the end

Xtest_name = args['f2']
Ytest_predict_name = args['o']

tree_name = args['t']



Xtrain, Ytrain, Xtest = load_data(Xtrain_name, Xtest_name)

dc_tree = ID3Algo.ID3Tree(pval)
root = dc_tree.trainTree(Xtrain, Ytrain)



print("Testing...")

Ypredict = dc_tree.predict_test_data(Xtest)
print "Ypredict is : ", Ypredict

# #generate random labels
TestPredict = []
for label in Ypredict:
    TestPredict.append([label])
with open(Ytest_predict_name, "wb") as f:
    writer = csv.writer(f)
    writer.writerows(TestPredict)

print("Output files generated")


def save_tree(root, filename):
    obj = open(filename, 'w')
    pkl.dump(root, obj)

save_tree(root, tree_name)

'''
train_feature, train_label, \
test_feature,  test_label = parseFiles.readData(trainFeature, trainLabel, testFeature, testLabel)
'''

# train_feature, train_label, \
# test_feature  = load_data(trainFeature,testFeature)
# print "finish reading data!"
#
# dc_tree = ID3Algo.ID3Tree(1)
# dc_tree.trainTree(train_feature, train_label)
#
#
# #dc_tree.trainTree(simple_feature, simple_label)
#
# predict, accuracy = dc_tree.predict_test_data(test_feature, test_label)
# print "Node in the decision tree is: %d" % dc_tree.numberOfNodes
#
# # print "Precision is: %.3f" %precision
# # print "Recall is: %.3f" %recall
# print "Accuracy is: %.3f" %accuracy