from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
import numpy as np
import pandas as pd
import os

'''
This function creates images for the minority class to compensate and have in the end the same amount of images for each class.
'''
def generate_minority_class_images():
    df = pd.read_csv("full_df.csv")
    df['labels'] = df['labels'].replace({"['N']": '0', "['D']": '1'}).astype(str)

    datagen = ImageDataGenerator(
        rescale=1./255,  
        rotation_range=20,  
        width_shift_range=0.2, 
        height_shift_range=0.2,  
        shear_range=0.2,  
        zoom_range=0.2, 
        horizontal_flip=True, 
        fill_mode='nearest' 
    )

    minority_class = df[df['labels'] == '1']
    save_dir = '/home/ander/Desktop/Cuarto/TFG/MinorityClassAugment'
    num_to_generate = len(df[df['labels'] == '0']) - len(minority_class)
    print(minority_class)

    generated = 0
    for _, row in minority_class.iterrows():
        img_path = '/home/ander/Desktop/Cuarto/TFG/Train/' + row['filename']
        img = load_img(img_path)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        for batch in datagen.flow(x, batch_size=1, save_to_dir=save_dir, save_prefix='aug', save_format='jpeg'):
            generated += 1
            if generated >= num_to_generate:
                break
        if generated >= num_to_generate:
            break


generate_minority_class_images()


