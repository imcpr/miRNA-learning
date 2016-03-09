#Author: Casper Liu

import numpy as np
import csv
import os
import cPickle

DATA_DIR = "/home/mcb/mlecle25/research/AncestralTargets/Data/targetgenes/analysis_miranda"

FILE_NAME = "/hsa-miR-6738-3p.miranda.sc140.csv"

n_files = 0
all_pos_labels = 0
all_labels = 0
taken_files = 0

X = []

for filename in os.listdir(DATA_DIR):
    n_files += 1
    print "%d file" % n_files
    with open(DATA_DIR+"/"+filename, "r") as handle:
        ii = 0
        pos_labels = 0
        F = []
        reader = csv.reader(handle)
        for row in reader:
            if ii == 0:
                ii += 1
                continue
            cols = row[0].split()
            if cols[-1] == '1':
                pos_labels += 1
            temp = [float(i) for i in cols[2:71]] + [float(cols[-1])]
            F.append(temp)
            ii += 1
        print "num of pos labels %d" % pos_labels
        print "num of genes in file %d" % ii
        if (pos_labels >= 500):
            print "File %d %s has over 500 pos sites, added" % (n_files, filename)
            all_labels += ii
            all_pos_labels += pos_labels
            taken_files += 1
            X.extend(F)
    
X = np.array(X)
print "There are %d pos labels out of %d labels in %d files , ratio %f" % (all_pos_labels,all_labels, taken_files, (float(pos_labels)/all_labels))
cPickle.dump(X, open("all_data.pkl", "wb"))