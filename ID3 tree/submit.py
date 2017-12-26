import argparse
import sys
sys.setrecursionlimit(10000)
import csv
import math
from scipy import stats
import pickle as pkl


'''
TreeNode represents a node in your decision tree
TreeNode can be:
    - A non-leaf node: 
        - data: contains the feature number this node is using to split the data
        - children[0]-children[4]: Each correspond to one of the values that the feature can take

    - A leaf node:
        - data: 'T' or 'F' 
        - children[0]-children[4]: Doesn't matter, you can leave them the same or cast to None.

'''


# DO NOT CHANGE THIS CLASS
class TreeNode():
    def __init__(self, data='T', children=[-1] * 5):
        self.nodes = list(children)
        self.data = data

    def save_tree(self, filename):
        obj = open(filename, 'w')
        pkl.dump(self, obj)


# loads Train and Test data
def load_data(ftrain, ftest):
    Xtrain, Ytrain, Xtest = [], [], []
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



class ID3Tree:
    def __init__(self, alpha):
        self.numberOfNodes = 0
        self.pvalueTreshold = alpha
        print 'Treshold', self.pvalueTreshold

    def trainTree(self, train, label):
        print len(train)
        usedFeature = [False for i in range(0, len(train[0]))]  # array of used feature initially all are 0
        self.root = self.constructTree(train, label, usedFeature)
        print 'Training complete'
        return self.root

    def constructTree(self, train, label, usedFeature):
        print len(train)
        majLabel, isSame = self.getMajorityLabel(label)
        self.numberOfNodes += 1
        '''majLabel is either 0 or 1'''
        # for each feature check the majority label and isSame
        # for root just check the label file, if for any feature labels are evenly distributed it means max entropy 1
        # should not use this feature and should return
        if False not in usedFeature:
            ''' all featues have been exhausted'''
            if majLabel == 0:
                root = TreeNode('F', [])
            else:
                root = TreeNode('T', [])
            return root
        elif isSame:
            if majLabel == 0:
                root = TreeNode('F', [])
            else:
                root = TreeNode('T', [])
            return root
        else:
            feature_index = self.selectNextFeature(train, label, usedFeature)
            if feature_index >= len(train[0]):
                print 'feature_index cannot be greater than  length of features'
            root = TreeNode(feature_index)

            splitted_dataset = {}  # a dictionary of unique feature values and their sub labels

            ''' get all the unique value of the this feature'''
            uniqueVals = []
            # now get the count of these unique vals in 274 labels
            for featureRow in train:
                v = featureRow[feature_index]
                if v not in uniqueVals:
                    uniqueVals.append(v)

            # for each unique val get the no. of positives and negatives
            for v in uniqueVals:
                sub_train, sub_label, new_used = self.splitDataSet(train, label, feature_index, v, usedFeature)
                splitted_dataset[v] = [sub_train, sub_label, new_used]

            # chi square test treshold code
            # use the chi-square test to test relevance
            # if irrelevant just stops here and uses majority as label
            num_pos = 0.0
            num_neg = 0.0
            for l in label:
                if l == 0:
                    num_neg += 1
                else:
                    num_pos += 1

            S = 0.0
            for k, v in splitted_dataset.iteritems():
                # k is the unique feature vaue
                # v[0] is the sub trainig set where k is present
                # v[1] is the sub label set where k is present
                p_exp = num_pos * len(v[1]) / float(len(train))
                n_exp = num_neg * len(v[1]) / float(len(train))
                real_p = 0.0
                real_n = 0.0
                for l in v[1]:
                    if l == 0:
                        real_n += 1
                    else:
                        real_p += 1
                # add to S
                tmp = 0
                if real_p != 0:
                    tmp += math.pow(p_exp - real_p, 2) / real_p

                if real_n != 0:
                    tmp += math.pow(n_exp - real_n, 2) / real_n

                S += tmp


            p_value = 1 - stats.chi2.cdf(S, len(splitted_dataset))

            print "chi-square p-value is: %.3f" % p_value

            ############################################################
            print p_value
            print self.pvalueTreshold
            cnt0 = 0.0
            cnt1 = 0.0

            if p_value < self.pvalueTreshold:
                l = len(splitted_dataset)
                for k, v in splitted_dataset.iteritems():
                    cnt0 += v[1].count(0)
                    cnt1 += v[1].count(1)

                    child = self.constructTree(v[0], v[1], v[2])
                    root.nodes[k-1] = child

                m = max(cnt0,cnt1)
                if m == cnt0:
                    m = 'F'
                else:
                    m = 'T'
                for  i in range(0,len(root.nodes)):
                    if root.nodes[i] == -1:
                        root.nodes[i] = TreeNode(m, [])

            else:
                if majLabel == 0:
                    root = TreeNode('F', [])
                else:
                    root = TreeNode('T', [])
                return root

        return root

    def splitDataSet(self, train, label, feature_index, v, usedFeature):
        count = 0
        sub_train = []
        sub_label = []

        newUsed = usedFeature[:]

        for row, l in zip(train, label):
            if row[feature_index] == v:
                count = count + 1
                sub_train.append(row)
                sub_label.append(l)

        newUsed[feature_index] = True
        return sub_train, sub_label, newUsed

    def getMajorityLabel(self, label):
        pos_label = 0.0
        neg_label = 0.0

        for l in label:
            if l == 0:
                neg_label += 1
            elif l == 1:
                pos_label += 1
            else:
                print 'label contains something other than 0 or 1'

        if pos_label == 0 or neg_label == 0:
            if pos_label > neg_label:
                return 1, True
            else:
                return 0, True
        else:
            if (neg_label / pos_label) > 1.0:
                return 0, False
            else:
                return 1, False

    def selectNextFeature(self, train, label, usedFeature):
        # choose the best feature which maximize entropy gain
        # used_feature is a list of bool of dimension of trainning set size

        n = len(train[0])
        ent = self.rootEntropy(label)
        max = -1
        max_index = -1

        # find the largest gain
        for i in range(0, n):
            # if this feature is no longer available
            if usedFeature[i]:
                continue

            # find largest gain
            f_entropy = self.featureEntropy(train, label, i)
            gain = ent - f_entropy
            if gain < -1e-10:
                print "Big error, gain smaller than 0"
                print gain
                exit()

            # print "%d feature gain is %.8f" %(i,gain)
            if gain > max:
                max = gain
                max_index = i

        # print "max gain is %.2f" %max
        return max_index

    def rootEntropy(self, label):
        pos_num = 0.0
        neg_num = 0.0
        total = len(label)
        for l in label:
            if l == 0:
                neg_num += 1
            elif l == 1:
                pos_num += 1

        entropy = 0.0
        if pos_num != 0:
            entropy -= (pos_num / total) * math.log(pos_num / total, 2)
        if neg_num != 0:
            entropy -= (neg_num / total) * math.log(neg_num / total, 2)

        return entropy

    def featureEntropy(self, training, label,feature_index):
        # If split on f_index attribute, how much gain can I get
        # collect how many value and their frequency
        val_freq = {}
        total = len(training)
        for feature in training:
            val = feature[feature_index]
            if val in val_freq:
                val_freq[val] += 1.0
            else:
                val_freq[val] = 1.0

        # now compute the entropy for each of this value
        # and average them by frequency
        f_entropy = 0.0
        for key, value in val_freq.iteritems():
            weight = value / total
            sublabel = []
            for t, l in zip(training, label):
                if t[feature_index] == key:
                    sublabel.append(l)
            f_entropy += weight * self.rootEntropy(sublabel)
        return f_entropy

    def predict_test_data(self, features):
        predict = []
        cnt = 0
        for f in features:
            cnt += 1

            test_label = self.result_value(f)
            if test_label == 'T':
                test = 1
            else:
                test = 0
            predict.append(test)

        return predict

    def result_value(self, test_data):

        node = self.root
        while len(node.nodes) != 0:
            f_index = node.data
            f_val = test_data[f_index]
            if node.nodes[f_val-1] == -1:
                    for nod in node.nodes:
                            if nod != -1:
                                    node = nod
                                    break
            else:
                node  = node.nodes[f_val - 1]
        return node.data


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
dc_tree = ID3Tree(pval)
root = dc_tree.trainTree(Xtrain, Ytrain)
print 'Number of nodes expanded is:',dc_tree.numberOfNodes
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










