import os

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

def processing(im_path,dimx,dimy):
    img = image.load_img(im_path, target_size=(dimx, dimy))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x
conv_out_train=[]
with tf.Graph().as_default():
    tf.logging.set_verbosity(tf.logging.INFO)


    images = tf.placeholder(tf.float32, shape=(1, dimx, dimy, 3))
    labels = tf.placeholder(tf.uint8, shape=(1,1))

    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        logits, _ = resnet_v1.resnet_v1_101(images, num_classes=num_classes, is_training=False)

    probs = tf.argmax(tf.nn.softmax(logits),axis=1)
    one_hot_labels = slim.one_hot_encoding(labels, num_classes)
    gt = tf.argmax(one_hot_labels,axis=-1)
    accuracy = slim.metrics.accuracy(probs, gt)
    init_fn = get_init_fn(model_path);
    with tf.Session() as sess:

        sess.run(tf.initialize_all_variables())
        init_fn(sess)

        #conv_out_train = []
        for class_id,prod_per_class in enumerate(product_files_per_class):
            conv_out = []
            for prod in prod_per_class:
                prod = prod.strip()
                print("Processing for {}".format(prod))
                img = processing(prod,dimx,dimy)
                feed_dict = {images:img,labels:np.array([[class_id]])}

                b4_conv3_out_t = tf.get_default_graph().get_tensor_by_name('resnet_v1_101/block4/unit_3/bottleneck_v1/Relu:0')
                out_t = tf.reshape(b4_conv3_out_t[0],(196,2048))
                out_t = tf.nn.l2_normalize(out_t,dim=-1)
                prob,acc,out = sess.run([probs,accuracy,out_t],feed_dict=feed_dict)
                print("Accuracy:{}".format(acc))
                
                #out = tf.nn.l2_normalize(out,dim=-1)
                print(out.shape)
                conv_out.append(out)
            conv_out_train.append(conv_out) 

for id,conv_out in enumerate(conv_out_train):

    filename = 'b4_conv3_out_class{}'.format(id) 
    with open(parent_dir+filename+'.pickle', 'wb') as handle:
        pickle.dump(np.array(conv_out), handle, protocol=pickle.HIGHEST_PROTOCOL)


