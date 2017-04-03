import tensorflow as tf
import numpy as np
import os
from PIL import Image
directory = "data/cards/"
original_width = 342
original_height = 245

x_dim = original_width//2
y_dim = original_height//2
n_filters = 4

dropout = .8


n_dims = 3


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


def convert_array_to_image(arr, disp=False):
    """
    takes an array of values -1 - 1 and converts it back to a
    pillow image file will display image if display is true
    """

    img_arr = ((arr + 1) * 128)
    img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_arr)
    if disp:
        img.show()
    return img


def autoencoder():

    w = {'conv1':tf.Variable(tf.random_normal([5,5,n_filters,32])),
               'conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               'fc':tf.Variable(tf.random_normal([43*31*64,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_dims])),
         'layer1': tf.Variable(tf.random_normal([n_dims, 1024])),
         'layer2': tf.Variable(tf.random_normal([1024, 2048])),
         'layer3': tf.Variable(tf.random_normal([2048, 512])),
         'layer4': tf.Variable(tf.random_normal([512,x_dim*y_dim*n_filters])),

                'convT': tf.Variable(tf.random_normal([x_dim*2,y_dim*2,n_filters,n_filters*2]))}

    b = {'conv1':tf.Variable(tf.random_normal([32])),
               'conv2':tf.Variable(tf.random_normal([64])),
               'fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_dims])),
                'layer1': tf.Variable(tf.random_normal([1024])),
         'layer2': tf.Variable(tf.random_normal([2048])),
         'layer3': tf.Variable(tf.random_normal([512])),
         'layer4': tf.Variable(tf.random_normal([x_dim * y_dim * n_filters]))}

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def maxpool2d(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    x = tf.placeholder(tf.float32, [1, x_dim, y_dim, n_filters])

    l1 = maxpool2d(conv2d(x, w['conv1']))
    l2 = tf.nn.relu(l1)

    l3 = maxpool2d(conv2d(l2, w['conv2']))
    l4= tf.nn.relu(l3)

    fc = tf.reshape(l4,[1, 43*31*64])
    fc = tf.nn.relu(tf.matmul(fc, w['fc'])+b['fc'])
    #fc = tf.nn.dropout(fc, dropout)

    output = tf.matmul(fc, w['out']) + b['out']

    l1 = tf.nn.softmax(tf.matmul(output, w['layer1']) + b['layer1'])
    l2 = tf.nn.softmax(tf.matmul(l1, w['layer2']) + b['layer2'])
    l3 = tf.nn.softmax(tf.matmul(l2, w['layer3']) + b['layer3'])
    output_connected = tf.matmul(l3, w['layer4']) + b['layer4']



    reshape_decode = tf.reshape(output_connected, [1,x_dim,y_dim,n_filters])

    autoencoded_output = reshape_decode
    #
    # autoencoded_output = tf.nn.conv2d_transpose(reshape_decode, w['convT'],
    #                                             output_shape=[1, x_dim, y_dim, n_filters],
    #                                             strides=[1, 2, 2, 1],
    #                                             padding='SAME',
    #                                             data_format='NHWC')

    cost = tf.reduce_sum(tf.square(autoencoded_output - x))

    opt = tf.train.AdamOptimizer(0.001).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        i = 0
        for filename in os.listdir(directory):
            try:
                img = load_image_values(directory+filename)
                feed_dict = {x: [img]}
                r_cost, out_img,_ = sess.run([cost,autoencoded_output,opt],feed_dict=feed_dict)
                print("Epoch " + str(i) + " Cost: " + str(r_cost))
                out_img = np.array(out_img).squeeze()
                if i % 100 == 0:
                    convert_array_to_image(arr=out_img,disp=True)
            except:
                print("Failed: " + filename)
            i += 1


autoencoder()