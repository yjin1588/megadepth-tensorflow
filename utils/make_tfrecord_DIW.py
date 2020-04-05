import tensorflow as tf

import csv
import os
from PIL import Image
import math

DATA_ROOT = '../../DIW_dataset/'
SAVE_ROOT = '../datasets/tfrecord/'
HEIGHT = 240
WIDTH = 320


def convert_and_save(mode):
    if mode == 'train':
        save_name = 'train_val.tfrecords'
        csv_path = '../DIW_Annotations/DIW_Annotations/DIW_train_val.csv'
    elif mode == 'test':
        save_name = 'test.tfrecords'
        csv_path = '../DIW_Annotations/DIW_Annotations/DIW_test.csv'
    else:
        raise ValueError

    num_examples, dataset = load_csv(csv_path)

    if not os.path.exists(SAVE_ROOT):
        os.makedirs(SAVE_ROOT)
    save_path = SAVE_ROOT + save_name
    with tf.python_io.TFRecordWriter(save_path) as writer:
        for idx in range(num_examples):
            img_path  = dataset['image'][idx][0][1:] #.tostring() #delete '.'
            #img = Image.open(img_path).resize((WIDTH,HEIGHT))
            #img_str = np.array(img).tostring()

            tar_feature = dataset['target'][idx][0]
            y_a, x_a, y_b, x_b, order, origin_width, origin_height = tar_feature.split(',')
            rel_y_a = float(y_a)/float(origin_height)
            rel_x_a = float(x_a)/float(origin_width)
            rel_y_b = float(y_b)/float(origin_height)
            rel_x_b = float(x_b)/float(origin_width)

            new_y_a = min(HEIGHT, max(1, math.floor(rel_y_a * HEIGHT)))
            new_x_a = min(WIDTH, max(1, math.floor(rel_x_a * WIDTH)))
            new_y_b = min(HEIGHT, max(1, math.floor(rel_y_b * HEIGHT)))
            new_x_b = min(WIDTH, max(1, math.floor(rel_x_b * WIDTH)))

            if (new_y_a == new_y_b) and (new_x_a == new_x_b):
                if new_y_a > 1:
                    new_y_a = new_y_a - 1
                else:
                    new_y_a = new_y_a + 1

            if order == '>': # a is farther than b
                y_A = new_y_b #A is front, B is back
                x_A = new_x_b
                y_B = new_y_a
                x_B = new_x_a
            elif order == '<': # b is farther than a
                y_A = new_y_a #A is front, B is back
                x_A = new_x_a
                y_B = new_y_b
                x_B = new_x_b
            else:
                raise ValueError
                
            feature = {
                #'image/encode': bytes_feature(img_str),
                'image/height': int64_feature(HEIGHT),
                'image/width': int64_feature(WIDTH),
                'depth/y_A': int64_feature(y_A),
                'depth/x_A': int64_feature(x_A),
                'depth/y_B': int64_feature(y_B),
                'depth/x_B': int64_feature(x_B),
            }

            features = tf.train.Features(feature=feature)
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())
    
    print('==>Finish saving tfrecords {}'.format(mode))

def load_csv(path: str) -> dict:
    '''
    - input:
        - path:
            type: str
            example: '../DIW_Annotations/DIW_Annotations/DIW_test.csv'
            help: csv path
    - output:
        - dataset:
            type: dict
            example: {'image': str, 'target': dict}

    - .csv file format
        - format:
            Image Filename
	        y_A, x_A, y_B, x_B, ordinal relationship, image width, image height
        - example:
            ./DIW_test/ad956b51882fce447aa5a5bb65e434366cb41aa1.thumb
            118,70,257,291,>,500,333
    '''

    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        reader_list = list(reader)
        images = reader_list[::2]

        depth_features = reader_list[1::2]
        assert(len(images) == len(depth_features))
        print('==> Finish loading csv file!')
        return len(images), {'image': images, 'target': depth_features}


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def main():
    convert_and_save('train')
    convert_and_save('test')

if __name__=='__main__':
    main()

