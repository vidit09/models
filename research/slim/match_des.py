import os
import glob

from datasets import flowers
from nets import resnet_v1
from nets import resnet_utils
from preprocessing import vgg_preprocessing
from keras import utils

from keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)
from keras.preprocessing import image
import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import pickle
import cv2
from cyvlfeat.vlad.vlad  import vlad
dimx = 448
dimy = 448
num_classes = 27

parent_dir = '/home/vidit/tfmodels/models/research/slim/train_resnet/'
model_path = parent_dir+'checkpoints/model2000.ckpt'
product_img_dir = '/home/vidit/Grocery_products/'
product_img_path = product_img_dir+'TrainingFiles.txt'
product_cat_exp = 'Training/Food'

cat_map = []
with open(product_img_dir+'cat_mapping.txt') as f:
    for line in f:
        cat_map.append(product_img_dir+line.strip())

with open(product_img_path) as f:
    train_files = f.readlines()

products = [p for p in filter(lambda x: product_cat_exp in x, train_files)]
product_files = []
for product in products:
    product_files.append(product_img_dir + product.strip().split('.')[0]+'_bkg_reduced.jpg')

product_files_per_class = [[p for p in product_files if cat in p] for cat in cat_map]
offset = np.cumsum([len(prod) for prod in product_files_per_class])
offset = np.hstack(([0],offset[:-1]))

def get_init_fn(model_path):
    """Returns a function run by the chief worker to warm-start the training."""
    checkpoint_exclude_scopes=[]

    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(
      model_path,
      variables_to_restore)

def processing(im_path,dimx,dimy,bbox):
    img = cv2.imread(im_path)
    crop = img[int(float(bbox[1])):int(float(bbox[3])),int(float(bbox[0])):int(float(bbox[2])),:]
    crop = cv2.resize(crop,(dimx, dimy))
    
    x = image.img_to_array(crop)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def get_vlad(kmeans,feat):
    centers = kmeans.cluster_centers_
    assgn = kmeans.predict(feat)


    assgn_mat = np.zeros((centers.shape[0],feat.shape[0]))
    for coor in zip(assgn,range(feat.shape[0])):
        assgn_mat[coor[0],coor[1]] = 1

    desc = vlad(feat.astype(np.float32).T, centers.astype(np.float32).T, assgn_mat.astype(np.float32), normalize_components=True)
    return desc
    

interbox = glob.glob('/home/vidit/intersection/*.txt')
VOCFormat = '/home/vidit/VOCFormat/Images/'
parent_dir = '/home/vidit/tfmodels/models/research/slim/train_resnet/'
kmeans_dir = parent_dir+'kmeans_out/'
vlad_dir = parent_dir+'vlad/'
n_cluster = 100
kmeans_basename = kmeans_dir+'kmean_out_class{}_{}.pkl'
vlad_basename = vlad_dir+'desc_class{}_{}.pkl'

final_out = 'out/'
if not os.path.exists(final_out):
    os.mkdir(final_out)

with tf.Graph().as_default():
    tf.logging.set_verbosity(tf.logging.INFO)


    images = tf.placeholder(tf.float32, shape=(1, dimx, dimy, 3))


    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        logits, _ = resnet_v1.resnet_v1_101(images, num_classes=num_classes, is_training=False)

    probs = tf.argmax(tf.nn.softmax(logits),axis=1)

    init_fn = get_init_fn(model_path)

    with tf.Session() as sess:

        sess.run(tf.initialize_all_variables())
        init_fn(sess)

        #conv_out_train = []
        for res in interbox:
            print('Processing file:{}'.format(res))
            imname = VOCFormat+'_'.join(os.path.basename(res).split('_')[4:])
            imname = imname.replace('txt','jpg')
            print(imname)
            with open(final_out+os.path.basename(res),'w') as ff:
                with open(res,'r') as rf:
                    for line in rf:

                        comp = line.split()
                        bbox = comp[:4]
                        label = comp[4].split('_')[1]
                        id = comp[5]

                        
                        img = processing(imname,dimx,dimy,bbox)
                        feed_dict = {images:img}

                        b4_conv3_out_t = tf.get_default_graph().get_tensor_by_name('resnet_v1_101/block4/unit_3/bottleneck_v1/Relu:0')
                        out_t = tf.reshape(b4_conv3_out_t[0],(196,2048))
                        out_t = tf.nn.l2_normalize(out_t,dim=-1)
                        prob,out = sess.run([probs,out_t],feed_dict=feed_dict)

                        with open(kmeans_basename.format(int(label)-1,n_cluster),'rb') as kf:
                            kmeans = pickle.load(kf)

                        desc = get_vlad(kmeans,out)

                        with open(vlad_basename.format(int(label)-1,n_cluster),'rb') as vf:
                            vlad_desc = pickle.load(vf)

                        dist = np.linalg.norm(desc - vlad_desc, axis=1)
                        nn = sorted(range(len(dist)), key=lambda k: dist[k])
                        
                        for iii in range(5):
                            #pre = product_files_per_class[int(label)-1][nn[iii]]
                            #print(pre,":",dist[nn[iii]])
                        
                            pred_label = product_files_per_class[int(label)-1][nn[iii]]
                            pred_label = os.path.basename(pred_label).replace('_bkg_reduced.jpg','')
                            pred_label = offset[int(label)-1]+int(pred_label)
                            ff.write(id+' '+str(pred_label)+'\n')



