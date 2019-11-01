import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

import sys
 
import tensorflow as tf
tf.app.flags.DEFINE_string('image_dir', None, 'gpus to use')
 

FLAGS = tf.app.flags.FLAGS


def xml_to_csv(path):
    xml_list = []
    
   
    for xml_file in glob.glob(path + '/*.xml'):
        
        print(xml_file)

        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size').find('width').text),
                     int(root.find('size').find('height').text),
                     str(member.find('name').text),
                     int(member.find('bndbox').find('xmin').text),
                     int(member.find('bndbox').find('ymin').text),
                     int(member.find('bndbox').find('xmax').text),
                     int(member.find('bndbox').find('ymax').text),
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    
    return xml_df


def main():
     

    print(FLAGS.image_dir)


    for folder in ['train','test']:
        image_path = os.path.join(os.getcwd(), ( FLAGS.image_dir + folder))
        print(image_path)

        xml_df = xml_to_csv(image_path)
        xml_df.to_csv(( FLAGS.image_dir + folder + '_labels.csv'), index=None)
        print('Successfully converted xml to csv.')


main()
