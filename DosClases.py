import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
############################################ READING AND CLEANING DATAFRAME ################################################

def Reading_csv():
    df = pd.read_csv("full_df.csv")
    # Realizar el filtrado y luego hacer una copia del DataFrame resultante para evitar SettingWithCopyWarning
    df_filtering = df[df['labels'].isin(["['N']", "['D']"])].copy()
    df_filtering.loc[:, 'labels'] = df_filtering['labels'].replace({"['N']": '0', "['D']": '1'}).astype(str)
    df_filtering = df_filtering[['filename', 'labels']]
    df_filtering = df_filtering.reset_index(drop=True)
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

    df_filtering['filename'] = '/Train/' + df_filtering['filename']

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
        history = model.fit(train_generator, steps_per_epoch=steps_per_epoch)
    return history

def plot_training_history(history):
    # Precisión (Accuracy)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    # Pérdida (Loss)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.show()


if __name__ == "__main__":
    df_filtering = Reading_csv()
    train_generator = preprocces_images(df_filtering)
    model = build_model()

    start_time = time.time()
    history = train_model(model, train_generator, 10, 100)
    end_time = time.time()
    plot_training_history(history)
    print(f"Total training time: {end_time - start_time} seconds")
