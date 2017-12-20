import glob
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.preprocessing import normalize
import pickle
import sys
import os

parent_dir = '/home/vidit/tfmodels/models/research/slim/train_resnet/'
kmeans_out_dir = parent_dir+'kmeans_out/'
if not os.path.exists(kmeans_out_dir):
    os.mkdir(kmeans_out_dir)


conv_out_basename = parent_dir+'b4_conv3_out_class'

for class_id in range(27):
    filename = conv_out_basename+str(class_id)+'.pickle'
    with open(filename,'rb') as f:
        conv_out = pickle.load(f)
    #print(conv_out.shape)
    conv_out =  np.reshape(conv_out,(conv_out.shape[0]*conv_out.shape[1],conv_out.shape[2]))
    print('Starting KMeans ...')
    for csize in [20,50,100]:
        k_means = MiniBatchKMeans(init='k-means++', n_clusters=csize, batch_size=10000,
                      n_init=5, max_no_improvement=200, verbose=1)
        k_means.fit(conv_out)
        print('Done for cluster sz:{} class{}'.format(csize,class_id))
        with open(kmeans_out_dir+'kmean_out_class'+str(class_id)+'_'+str(csize)+'.pkl','wb') as f:
            pickle.dump(k_means,f)

