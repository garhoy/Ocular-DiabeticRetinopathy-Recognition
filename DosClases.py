import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, DenseNet121, EfficientNetB7
from tensorflow.keras import layers, models
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam

############################################ READING AND CLEANING DATAFRAME #############################################
def Reading_csv():
    df = pd.read_csv("full_df.csv")

    df_filtering = df[df['labels'].isin(["['N']", "['D']"])].copy()
    df_filtering.loc[:, 'labels'] = df_filtering['labels'].replace({"['N']": '0', "['D']": '1'}).astype(str)
    df_filtering = df_filtering[['filename', 'labels']]
    df_filtering = df_filtering.reset_index(drop=True)
    print("LABELS :")
    print(df_filtering['labels'].value_counts())
    return df_filtering

########################################## PREPROCESSING IMAGES #########################################################


def preprocces_images(df_filtering):
    global target_size
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

    df_filtering['filename'] = '/home/ander/Desktop/Cuarto/TFG/Train/' + df_filtering['filename']
    print(df_filtering)
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

def add_new_images_to_df(df, start_number, num_new_images, label):
    new_entries = []
    file_number = start_number
    is_right = True

    for i in range(num_new_images):
        side = '_right' if is_right else '_left'
        filename = f"{file_number}{side}.jpg"
        new_entries.append({'filename': filename, 'labels': label})

        if is_right:
            file_number += 1
        is_right = not is_right

    new_df = pd.DataFrame(new_entries)
    return pd.concat([df, new_df], ignore_index=True)

def build_model(fine_tune_layers = 0): 
    global input_shape
    model_name = "Resnet"

    if   model_name == "Resnet":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == "Inception":
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == "DenseNet":
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == "EfficientNet":
        base_model = EfficientNetB7(weights='imagenet', include_top=False, input_shape=input_shape)
    
    for layer in base_model.layers:
        layer.trainable = False


    if fine_tune_layers > 0:
        for layer in base_model.layers[-fine_tune_layers:]:
            layer.trainable = True


    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    predictions = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    return model



def train_model(model, train_generator, epochs, steps_per_epoch):
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        history = model.fit(train_generator, steps_per_epoch=steps_per_epoch)
    return history

def plot_training_history(history):

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()


    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig("Training_history.png")
    plt.show()


if __name__ == "__main__":
    df_filtering = Reading_csv()
    start_number = 4785 
    num_new_images = 1265
    target_size = (512,512)
    input_shape = (512,512,3)
    df_filtering = add_new_images_to_df(df_filtering, start_number, num_new_images, '1')
    print("Labels value count :")
    print(df_filtering['labels'].value_counts())
    train_generator = preprocces_images(df_filtering)
    model = build_model(5)

    start_time = time.time()
    history = train_model(model, train_generator, 15, 100)
    end_time = time.time()
    plot_training_history(history)
    print(f"Total training time: {end_time - start_time} seconds")