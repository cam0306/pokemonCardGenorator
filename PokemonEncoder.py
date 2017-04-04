import tensorflow as tf
import numpy as np
import os
from PIL import Image
import random
directory = "data/cards/"
original_width = 342
original_height = 245

x_dim = original_width//2
y_dim = original_height//2
n_filters = 4

dropout = .8


n_dims = 512

# hyper params
training_epochs = 100000
learning_rate = 0.001

# User settings
IMAGE_DISPLAY_RATE = 50         # rate at which cards will display
DISPLAY_IMAGES = True           # show cards at epochs
TRAIN = True
CONTINUE_FROM_CHECKPOINT = False
cardToTry = "22_Aggron.png"
batch_size = 1

# convolution params
conv1_channels = 32
conv1_filter_size = 4

conv2_channels = 32
conv2_filter_size = 4

conv3_channels = 32
conv3_filter_size = 4

conv4_channels = 32
conv4_filter_size = 4

conv5_channels = 32
conv5_filter_size = 4

conv6_channels = 32
conv6_filter_size = 4


def load_image_values(location):
    """
    location is the location of the data
    converts an image file into a numpy array
    """
    im = Image.open(location)
    im = im.resize((y_dim, x_dim))
    lst = np.array(im)
    lst = (lst / 128) - 1
    return lst


def convert_array_to_image(arr):
    """
    takes an array of values -1 - 1 and converts it back to a
    pillow image file will display image if display is true
    """

    img_arr = ((arr + 1) * 128)
    img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_arr)

    return img

def merge_images(file1, file2):
    """Merge two images into one, displayed side by side
    :param file1: path to first image file
    :param file2: path to second image file
    :return: the merged Image object
    """
    image1 = file1
    image2 = file2

    (width1, height1) = image1.size
    (width2, height2) = image2.size

    result_width = width1 + width2
    result_height = max(height1, height2)

    result = Image.new('RGB', (result_width, result_height))
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(width1, 0))
    return result


def unpool(value, name='unpool'):
    """N-dimensional version of the unpooling operation from
    https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf

    :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
    :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
    """
    with tf.name_scope(name) as scope:
        sh = value.get_shape().as_list()
        dim = len(sh[1:-1])
        out = (tf.reshape(value, [-1] + sh[-dim:]))
        for i in range(dim, 0, -1):
            out = tf.concat(i, [out, tf.zeros_like(out)])
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size, name=scope)
    return out


def deconv2d(x, W):
    return tf.nn.conv2d_transpose(x, W, strides=[1, 1, 1, 1], padding='SAME')


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def autoencoder(x):

    w = {'conv1': tf.Variable(tf.random_normal([conv1_filter_size, conv1_filter_size, n_filters, conv1_channels])),
         # RELU
         'conv2': tf.Variable(tf.random_normal([conv2_filter_size, conv2_filter_size, conv1_channels, conv2_channels])),
         # RELU
         # Pool
         'conv3': tf.Variable(tf.random_normal([conv3_filter_size, conv3_filter_size, conv2_channels, conv3_channels])),
         # RELU
         'conv4': tf.Variable(tf.random_normal([conv4_filter_size, conv4_filter_size, conv3_channels, conv4_channels])),
         # RELU
         # POOL
         'conv5': tf.Variable(tf.random_normal([conv5_filter_size, conv5_filter_size, conv4_channels, conv5_channels])),
         # RELU
         'conv6': tf.Variable(tf.random_normal([conv6_filter_size, conv6_filter_size, conv5_channels, conv5_channels])),
         # RELU
         # POOL
         # fully connected
         }

    b = {'conv1':tf.Variable(tf.random_normal([conv1_channels])),
         'conv2':tf.Variable(tf.random_normal([conv2_channels])),
         'conv3': tf.Variable(tf.random_normal([conv3_channels])),
         'conv4': tf.Variable(tf.random_normal([conv4_channels])),
         'conv5': tf.Variable(tf.random_normal([conv5_channels])),
         'conv6': tf.Variable(tf.random_normal([conv6_channels])),

         'encode': tf.Variable(tf.random_normal([n_dims]))
         }




    l1 = tf.nn.relu(maxpool2d(conv2d(x, w['conv1'])) + b['conv1'])

    l2 = tf.nn.relu(maxpool2d(conv2d(x, w['conv1'])) + b['conv1'])
    l2 = maxpool2d(l2)

    l3 = tf.nn.relu(maxpool2d(conv2d(x, w['conv1'])) + b['conv1'])

    l4 = tf.nn.relu(maxpool2d(conv2d(x, w['conv1'])) + b['conv1'])
    l4 = maxpool2d(l4)

    l5 = tf.nn.relu(maxpool2d(conv2d(x, w['conv1'])) + b['conv1'])

    l6 = tf.nn.relu(maxpool2d(conv2d(x, w['conv1'])) + b['conv1'])
    l6 = maxpool2d(l6)

    fully_connected_shape = l6.get_shape()
    with open("options.txt", 'w')as f:
        write_string = "Fully Connected Shape: "
        for i in fully_connected_shape:
            write_string += str(i) + ","
        f.write(write_string)

    fully_connected_size = sum(l6.get_shape())/batch_size

    w['encode'] = tf.Variable(tf.random_normal([fully_connected_size, n_dims]))
    fc = tf.reshape(l6, [batch_size, fully_connected_size])

    reduct_layer = tf.nn.relu(tf.matmul(fc, w['encode'])+b['encode'])

    return reduct_layer

def autodecoder(inputs):
    """
    inputs to decode
    :param inputs: tensor size [batchsize,n_dims]
    :return: output size of original image
    """
    # Load in the shape of the last layer of the encoder
    fully_connected_shape = []
    with open("options.txt",'r')as f:
        for line in f:
            if "Fully Connected Shape: " in line:
                line = line.replace("Fully Connected Shape: ","")
                for s in line.split(','):
                    fully_connected_shape += [int(s)]




    l1 = tf.nn.relu(maxpool2d(conv2d(x, w['conv1'])) + b['conv1'])

    l2 = tf.nn.relu(maxpool2d(conv2d(x, w['conv1'])) + b['conv1'])
    l2 = maxpool2d(l2)

    l3 = tf.nn.relu(maxpool2d(conv2d(x, w['conv1'])) + b['conv1'])

    l4 = tf.nn.relu(maxpool2d(conv2d(x, w['conv1'])) + b['conv1'])
    l4 = maxpool2d(l4)

    l5 = tf.nn.relu(maxpool2d(conv2d(x, w['conv1'])) + b['conv1'])

    l6 = tf.nn.relu(maxpool2d(conv2d(x, w['conv1'])) + b['conv1'])
    l6 = maxpool2d(l6)


def optim(x,y):
    """
    runs backpropagation through algorithm
    :param x: label image
    :param y: calculated image
    :return: optimizer, calculated cost
    """
    cost = tf.reduce_sum(tf.square(y - x))

    opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    return opt, cost


x = tf.placeholder(tf.float32, [1, x_dim, y_dim, n_filters])
reduction = autoencoder(x)
y = autodecoder(reduction)
opt = optim(x,y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    i = 0
    saver = tf.train.Saver()
    if TRAIN:
        if CONTINUE_FROM_CHECKPOINT:
            new_saver = tf.train.import_meta_graph('checkpoints/pokemon_model.meta')
            new_saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))

        while i < training_epochs:
            for filename in os.listdir(directory):
                try:
                    img = load_image_values(directory+filename)
                    feed_dict = {x: [img]}
                    r_cost, out_img,_ = sess.run([cost, autoencoded_output, opt], feed_dict=feed_dict)

                    out_img = np.array(out_img).squeeze()

                    if i % 10 == 0:
                        print("Epoch " + str(i) + " Cost: " + str(r_cost))

                    if i % IMAGE_DISPLAY_RATE == 0 and DISPLAY_IMAGES:
                        processed_image = convert_array_to_image(arr=out_img)
                        original_image = convert_array_to_image(img)
                        combined_image = merge_images(original_image, processed_image)
                        combined_image.show()

                    if i % 1000 == 0:
                        path = saver.save(sess, 'checkpoints/pokemon_model')
                        print("model saved: " + path)

                except OSError:
                    print("Failed to load: " + filename)
                except ValueError:
                    print("incorrect image size: " + filename)
                i += 1
    else:
        new_saver = tf.train.import_meta_graph('checkpoints/pokemon_model.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
        files = []
        for filename in os.listdir(directory):
            files += [directory + filename]
        random.shuffle(files)
        for f in files:
            try:
                img = load_image_values(f)
                feed_dict = {x: [img]}
                out_img = sess.run([autoencoded_output], feed_dict=feed_dict)
                out_img = np.array(out_img).squeeze()
                processed_image = convert_array_to_image(arr=out_img)
                original_image = convert_array_to_image(img)
                combined_image = merge_images(original_image, processed_image)
                combined_image.show()
            except OSError:
                print("Failed to load: " + filename)
            except ValueError:
                print("incorrect image size: " + filename)
            input("press Enter to view next ")



