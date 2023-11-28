from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
import numpy as np
import pandas as pd
import os
from PIL import Image
'''
This function creates images for the minority class to compensate and have in the end the same amount of images for each class.
'''
def generate_minority_class_images(csv_file, save_dir, start_number):
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

    num_to_generate = len(df[df['labels'] == '0']) - len(minority_class)
    generated = 0
    file_number = start_number
    is_right = True  

    for _, row in minority_class.iterrows():
        img_path = '/home/ander/Desktop/Cuarto/TFG/Train/' + row['filename']
        img = load_img(img_path)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        for batch in datagen.flow(x, batch_size=1):
            image = batch[0]
            side = '_right' if is_right else '_left'
            new_filename = f"{file_number}{side}.jpg"
            new_filepath = os.path.join(save_dir, new_filename)
            Image.fromarray((image * 255).astype(np.uint8)).save(new_filepath)
            
            generated += 1
            if is_right:  
                file_number += 1
            is_right = not is_right

            if generated >= num_to_generate:
                break
        if generated >= num_to_generate:
            break

csv_file = "full_df.csv"
save_dir = '/home/ander/Desktop/Cuarto/TFG/Train'
start_number = 4785
generate_minority_class_images(csv_file, save_dir, start_number)

