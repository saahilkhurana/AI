import math
from scipy import stats
import pickle as pkl


class ID3Tree:
    def __init__(self, alpha):
        self.numberOfNodes = 0
        self.pvalueTreshold = alpha

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
                print 'Big Error, feature_index cannot be greater than  length of features'
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
                r_pi = 0.0
                r_ni = 0.0
                for l in v[1]:
                    if l == 0:
                        r_ni += 1
                    else:
                        r_pi += 1
                # add to S
                tmp = 0
                if r_pi != 0:
                    tmp += math.pow(p_exp - r_pi, 2) / r_pi

                if r_ni != 0:
                    tmp += math.pow(n_exp - r_ni, 2) / r_ni

                S += tmp

            # compute the p_value by scipys
            p_value = 1 - stats.chi2.cdf(S, len(splitted_dataset))

            print "chi-square p-value is: %.3f" % p_value

            ############################################################

            if p_value < self.pvalueTreshold:
                for k, v in splitted_dataset.iteritems():
                    child = self.constructTree(v[0], v[1], v[2])
                    root.nodes[k-1] = child
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
            # in training set there are more negative set than positive example
            if (neg_label / pos_label) > (32193.0 / 7807.0):
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
                print gain
                exit()

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
            else:
                print 'Incorrect value is label'

        entropy = 0.0
        if pos_num != 0:
            entropy -= (pos_num / total) * math.log(pos_num / total, 2)
        if neg_num != 0:
            entropy -= (neg_num / total) * math.log(neg_num / total, 2)

        return entropy

    def featureEntropy(self, train, label,f_index):
        # If split on f_index attribute, how much gain can I get
        # collect how many value and their frequency
        val_freq = {}
        total = len(train)
        for feature in train:
            val = feature[f_index]
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
            for f, l in zip(train, label):
                if f[f_index] == key:
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

        result_values = []

        # print("test data is : ", test_data)
        # for test_value in test_data:

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
            # result_values.append(node.predictLabel())

        return node.data

class TreeNode():
    def __init__(self, data='T', children=[-1] * 5):
            self.nodes = list(children)
            self.data = data

    def save_tree(self, filename):
            obj = open(filename, 'w')
            pkl.dump(self, obj)




