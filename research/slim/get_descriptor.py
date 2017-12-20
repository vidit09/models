import numpy as np
import os
from cyvlfeat.vlad.vlad  import vlad
import pickle

n_clusters = [100]

parent_dir = '/home/vidit/tfmodels/models/research/slim/train_resnet/'
kmeans_dir = parent_dir+'kmeans_out/'

vlad_out_dir = parent_dir+'vlad/'
if not os.path.exists(vlad_out_dir):
    os.mkdir(vlad_out_dir)

feat_basename = parent_dir+'b4_conv3_out_class{}'
kmeans_basename = kmeans_dir+'kmean_out_class{}_{}'

for class_id in range(27):
    print('For class {}'.format(class_id))
    feat_name = feat_basename.format(class_id)
    with open(feat_name+'.pickle','rb') as f:
        feat = pickle.load(f)

    feat_shape = feat.shape
    _feat = np.reshape(feat,(feat.shape[0]*feat.shape[1],feat.shape[2]))
    
    kmeans_name = kmeans_basename.format(class_id,n_clusters[0])
    with open(kmeans_name+'.pkl','rb') as f:
        kmean = pickle.load(f)

    centers = kmean.cluster_centers_
    assgn = kmean.predict(_feat)


    assgn_mat = np.zeros((centers.shape[0],_feat.shape[0]))
    for coor in zip(assgn,range(_feat.shape[0])):
        assgn_mat[coor[0],coor[1]] = 1
    enc = []
    for i in range(feat_shape[0]):
        sub_feat = feat[i,:,:]
        assgn_submat = assgn_mat[:,i*feat_shape[1]:(i+1)*feat_shape[1]]
        desc = vlad(sub_feat.astype(np.float32).T, centers.astype(np.float32).T, assgn_submat.astype(np.float32), normalize_components=True)
        enc.append(desc)
   
    enc = np.array(enc)
    print('Final desc shape{}'.format(enc.shape))

    with open(vlad_out_dir+'desc_class{}_{}.pkl'.format(class_id,n_clusters[0]),'wb') as f:
       pickle.dump(enc,f)
