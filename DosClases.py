import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
import tqdm 
import time 
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

def build_model(): 
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    predictions = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model



def train_model(model, train_generator, epochs, steps_per_epoch):
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        model.fit(train_generator, steps_per_epoch=steps_per_epoch)

if __name__ == "__main__":
    df_filtering = Reading_csv()
    train_generator = preprocces_images(df_filtering)
    model = build_model()

    start_time = time.time()
    train_model(model, train_generator, 10, 100)
    end_time = time.time()

    print(f"Total training time: {end_time - start_time} seconds")
