import os

from nets import resnet_v1
from keras import utils

from keras.applications.vgg16 import (preprocess_input)
from keras.preprocessing import image
import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np


class CustomDataGen():

    def __init__(self, dim_x, dim_y, dim_z, num_class, batch_size):
        self.batch_size = batch_size
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.num_class = num_class
        # self.augmentation = image.ImageDataGenerator(
        #     rotation_range=20,
        #     shear_range=0.5
        # )

    def randomize_ind(self,data):
        indexes = np.arange(len(data))
        np.random.shuffle(indexes)
        return indexes

    def get_data(self,list):

        X = np.empty((self.batch_size, self.dim_x, self.dim_y, self.dim_z))
        y = np.empty((self.batch_size,self.num_class))

        for id, data in enumerate(list):
            im_path = data.split(' ')[0]
            label = int(data.split(' ')[1])
            img = image.load_img(im_path, target_size=(self.dim_x, self.dim_y))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)[0]
            X[id,:,:,:] = x

            y_ = utils.to_categorical(label, self.num_class)
            y[id,...] = y_

        return X, y

    def generate_batch(self, data):

        while 1:
            indexes = self.randomize_ind(data)

            num_batch = int(len(indexes)/self.batch_size)
            for batch_id in range(num_batch):
                temp_list = [data[k] for k in indexes[batch_id*self.batch_size:(batch_id+1)*self.batch_size]]

                X,y = self.get_data(temp_list)
                # return self.augmentation.flow(X,y,self.batch_size)
                yield X,y

def read_img_list_from_file(directory,file_path):

    data = []
    with open(file_path) as f:
        for line in f:
            data.append(directory+line)

    return data

num_classes = 27
batch_size = 30
train_data = read_img_list_from_file('train_resnet/','train_resnet/train.txt')
val_data = read_img_list_from_file('train_resnet/','train_resnet/val.txt')
training_generator = CustomDataGen(224, 224, 3, num_classes, batch_size).generate_batch(train_data)
val_generator = CustomDataGen(224, 224, 3, num_classes, len(val_data)).generate_batch(val_data)

image_size = resnet_v1.resnet_v1_101.default_image_size

train_dir = 'train_resnet/'
ckpt_dir = 'train_resnet/'+'checkpoints'
log_dir = train_dir+'logs'
max_iters = 10000
save_ckpt_interval = 500
summary_interval = 200

def get_init_fn():
    """Returns a function run by the chief worker to warm-start the training."""
    checkpoint_exclude_scopes = ["resnet_v1_101/logits"]

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
        os.path.join(train_dir, 'resnet_v1_101.ckpt'),
        variables_to_restore)


with tf.Graph().as_default():
    tf.logging.set_verbosity(tf.logging.INFO)

    images = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    labels = tf.placeholder(tf.float32, shape=(None, 27))

    # Create the model, use the default arg scope to configure the batch norm parameters.
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        logits, _ = resnet_v1.resnet_v1_101(images, num_classes=num_classes, is_training=True)

    # Specify the loss function:
    slim.losses.softmax_cross_entropy(logits, labels)
    total_loss = slim.losses.get_total_loss()
    
    pred_class = tf.argmax(tf.nn.softmax(logits),axis=-1)
    gt_class = tf.argmax(labels,axis=-1)
    accuracy = slim.metrics.accuracy(pred_class, gt_class)
    
    # Create some summaries to visualize the training process:
    tf.summary.scalar('losses/Total_Loss', total_loss)
    tf.summary.scalar('acc', accuracy)
    merged = tf.summary.merge_all()
    

    # Specify the optimizer and create the train op:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_only = ['block4','logits']
    trainable = []
    for var in tf.trainable_variables():
        if any(prefix in var.name for prefix in train_only):
            print('Training var:{}'.format(var.name))
            trainable.append(var)

    train_op = slim.learning.create_train_op(total_loss, optimizer,variables_to_train=trainable)
    init_fn = get_init_fn();
    iters = 0
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        init_fn(sess)
        train_writer = tf.summary.FileWriter(log_dir + '/train',sess.graph)
        test_writer = tf.summary.FileWriter(log_dir + '/test')

        for b_images, b_labels in training_generator:

            if iters > max_iters:
                save_path = saver.save(sess, ckpt_dir+"/model-final.ckpt") 
                print('Saving Final Checkpoint:{}'.format(save_path))               
                break		
            if iters>0 and iters%save_ckpt_interval == 0:
                save_path = saver.save(sess, ckpt_dir+"/model"+str(iters)+".ckpt")
                print('Saving Checkpoint:{}'.format(save_path))

            feed_dict = {images: b_images, labels: b_labels}
            summary,final_loss, _, acc = sess.run([merged,total_loss, train_op, accuracy], feed_dict=feed_dict)
            train_writer.add_summary(summary, iters)
            
            if iters>0 and iters%summary_interval == 0:
                val_generator = CustomDataGen(224, 224, 3, num_classes, 200).generate_batch(val_data)
                val_batch = 0
                print('Running Validation')

                for v_images, v_labels in val_generator:
                    feed_dict = {images: v_images, labels: v_labels}
                    summary,final_loss = sess.run([merged,total_loss], feed_dict=feed_dict)
                    test_writer.add_summary(summary, iters+val_batch)
                    val_batch += 1
                    if val_batch == len(val_data)//200:
                        break

            print('For batch:{} loss:{} acc:{}'.format(iters,final_loss,acc))
            iters += 1
