import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

############################################ READING AND CLEANING DATAFRAME ################################################

def Reading_csv():
    df = pd.read_csv("full_df.csv")
    df_filtering = df[df['labels'].isin(["['N']", "['D']"])]
    df_filtering['labels'] = df_filtering['labels'].replace({"['N']": 0, "['D']": 1})
    
    df_filtering = df_filtering[['filename', 'labels']]
    df_filtering = df_filtering.reset_index(drop=True)
    df_filtering['labels'] = df_filtering['labels'].astype(str)
    return df_filtering

########################################## PREPROCESSING IMAGES #########################################################

def preprocces_images(df_filtering):
    # Target size needed for VGG16
    target_size = (224, 224)

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

    df_filtering['filename'] = 'Train/' + df_filtering['filename']

    train_generator = datagen.flow_from_dataframe(
        dataframe=df_filtering,
        directory='.',  
        x_col='filename',  
        y_col='labels',  
        target_size=target_size,  
        batch_size=32,
        class_mode='binary' 
    )

    return train_generator

if __name__ == "__main__":
    df_filtering = Reading_csv()
    preprocces_images(df_filtering)