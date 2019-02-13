import numpy as np
import os
import tensorflow as tf
import urllib
import matplotlib.pyplot as plt

from datasets import imagenet
from nets import inception
from preprocessing import inception_preprocessing

slim = tf.contrib.slim

batch_size = 3
image_size = inception.inception_v3.default_image_size

checkpoints_dir = '/root/code/model'
checkpoints_filename = 'inception_resnet_v2_2016_08_30.ckpt'
model_name = 'InceptionResnetV2'
sess = tf.InteractiveSession()
graph = tf.Graph()
graph.as_default()

def classify_from_url(url):
    image_string = urllib.urlopen(url).read()
    image = tf.image.decode_jpeg(image_string, channels=3)
    processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
    processed_images  = tf.expand_dims(processed_image, 0)
    
    # Create the model, use the default arg scope to configure the batch norm parameters.
    with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
        logits, _ = inception.inception_resnet_v2(processed_images, num_classes=1001, is_training=False)
    probabilities = tf.nn.softmax(logits)
    
    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, checkpoints_filename),
        slim.get_model_variables(model_name))
    
    init_fn(sess)
    np_image, probabilities = sess.run([image, probabilities])
    probabilities = probabilities[0, 0:]
    sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]
        
    plt.figure()
    plt.imshow(np_image.astype(np.uint8))
    plt.axis('off')
    plt.show()

    names = imagenet.create_readable_names_for_imagenet_labels()
    for i in range(5):
        index = sorted_inds[i]
        print('Probability %0.2f%% => [%s]' % (probabilities[index], names[index]))