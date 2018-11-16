import subprocess
import numpy as np 
from numpy import genfromtxt

#bashCommand = "java -cp ~/Downloads/weka-3-8-2/weka.jar weka.core.converters.CSVSaver -i eurlex_nA-5k_CV1-10_train.arff > eurlex_nA-5k_CV1-10_train.csv"
#process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
#output, error = process.communicate()

with open('eurlex_nA-5k_CV1-10_train.csv') as f:
    lines = f.read().splitlines()[0]

a = genfromtxt('eurlex_nA-5k_CV1-10_train.csv', delimiter=',')
words = lines.split(',')[1:]
doc_id = {}
doc_id_inv = {}

words_dict = {}
for i, w in enumerate(words):
    words_dict[i] = w

with open('feature_names.txt', 'w') as f:
    for key, value in words_dict.items():
        f.write('%s:%s\n' % (key, value))

for i in range(1, len(a[:,0])):
    doc_id_inv[a[i,0]] = i-1
    doc_id[i-1] = a[i,0]
# doc_id_list = doc_id.
x_tr = a[1:,1:]
np.save('words',words)
np.save('doc_id',doc_id) # dictionary
np.save('doc_id_inv',doc_id_inv) # dictionary
np.save('x_tr',x_tr)


labels_data_pt = genfromtxt('/u/79/wa.saxenas2/unix/Downloads/eurlex_id2class/id2class_eurlex_eurovoc.qrels', delimiter=' ')[:,1]
with open('/u/79/wa.saxenas2/unix/Downloads/eurlex_id2class/id2class_eurlex_eurovoc.qrels') as f:
    lines = f.read().splitlines()

label_names = []
for line in lines:
    label_names.append(line.split(' ')[0])


label_set = {}
label_set_inv = {}
count = 0
# data_map = {}
# data_count = 0
for i in range(np.shape(labels_data_pt)[0]):
    if label_names[i] not in label_set.keys():
        label_set[label_names[i]] = count
        label_set_inv[count] = label_names[i]        
        count+=1
        print(count)
    # if labels[i] not in data_map.keys() and labels[i] in doc_id_list:
    #     data_map[labels[i]] = data_count
    #     data_count+=1

np.save('label_set', label_set) # dictionary
np.save('label_set_inv', label_set_inv) # dictionary

with open('label_set.txt', 'w') as f:
    for key, value in label_set_inv.items():
        f.write('%s:%s\n' % (key, value))

y_tr = np.zeros((np.shape(x_tr)[0], count))
y_tr_named = {}
for i in range(np.shape(labels_data_pt)[0]):
    if labels_data_pt[i] in doc_id_inv.keys():
        y_tr[doc_id_inv[labels_data_pt[i]], label_set[label_names[i]]] = 1
        if doc_id_inv[labels_data_pt[i]] not in y_tr_named.keys():
            y_tr_named[doc_id_inv[labels_data_pt[i]]] = []
        y_tr_named[doc_id_inv[labels_data_pt[i]]].append(label_names[i])
np.save('y_tr', y_tr)

with open('y_tr_named.txt', 'w') as f:
    for key, value in y_tr_named.items():
        f.write('%s:%s\n' % (key, value))
