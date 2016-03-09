import numpy as np
import cPickle
import random


#36248 positives

# all = cPickle.load(open('all_data.pkl', 'rb'))

pos = []
neg = []

while len(pos) < 36248:
    r = random.randint(0, 717929)
    if all[r][-1] == 1:
        pos.append(all[r])

while len(neg) < 36248:
    r = random.randint(0, 717929)
    if all[r][-1] == 0:
        neg.append(all[r])

print len(pos)
print len(neg)

new = np.vstack((pos,neg))

np.random.shuffle(new)

cPickle.dump(new, open('5050_sampled_data.pkl', 'wb'))