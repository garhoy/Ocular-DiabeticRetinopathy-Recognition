print(__doc__)
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet152, InceptionV3, DenseNet121, EfficientNetB7
from tensorflow.keras import layers, models
from tqdm import tqdm
import time
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import itertools
import numpy as np
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.nan)
from tensorflow.keras.models import load_model
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

############################################ READING AND CLEANING DATAFRAME #############################################
def Reading_csv():
    df = pd.read_csv("/home/angarcia/datos/data/full_df.csv")

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
        fill_mode='nearest',
        validation_split=0.2  # por ejemplo, 20% para validación
    )

    df_filtering['filename'] = '/home/angarcia/datos/data/AugmentedDataTrain/' + df_filtering['filename']

    train_generator = datagen.flow_from_dataframe(
        dataframe=df_filtering,
        directory='.',
        x_col='filename',
        y_col='labels',
        target_size=target_size,
        batch_size=32,
        class_mode='binary',
        subset='training'
    )
    
    validation_generator = datagen.flow_from_dataframe(
        dataframe=df_filtering,
        directory='.',
        x_col='filename',
        y_col='labels',
        target_size=target_size,
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )

    return train_generator, validation_generator

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
    model_name = "DenseNet"

    if   model_name == "Resnet":
        base_model = ResNet152(weights='imagenet', include_top=False, input_shape=input_shape)
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


def train_model(model, train_generator, epochs, steps_per_epoch, validation_data):

    model_checkpoint_callback = ModelCheckpoint(
    filepath='/home/angarcia/datos/data/Modelo_DenseNetFine.h5',
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

    history = model.fit(
        train_generator,
        steps_per_epoch = steps_per_epoch,
        epochs=epochs,
        validation_data=validation_data,
        verbose=1,
        callbacks=[model_checkpoint_callback]  # Añade el callback aquí
    )
    return history


def plot_training_history(history, save_loss=True, save_accuracy=True):
    # Save the accuracy graph
    if save_accuracy:
        plt.figure(figsize=(6, 4))
        plt.plot(history.history['accuracy'], label='Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig("Accuracy_comparison.png")


    # Save the loss graph
    if save_loss:
        plt.figure(figsize=(6, 4))
        plt.plot(history.history['loss'], label='Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig("Loss_comparison.png")


def get_predictions_and_labels(model, generator):
    predictions = model.predict(generator, steps=np.ceil(generator.samples/generator.batch_size))
    y_score = (predictions > 0.5).astype('int').flatten()
    y_test = generator.classes
    return y_score, y_test

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, format(cm[i, j], '.2f'), 
                     horizontalalignment="center", 
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, format(cm[i, j], 'd'), 
                     horizontalalignment="center", 
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("Confusion_matrix.png")
    plt.close()


if __name__ == "__main__":
    df_filtering = Reading_csv()
    start_number = 4785
    num_new_images = 1265
    target_size = (512,512)
    input_shape = (512,512,3)

    model_path = '/home/angarcia/datos/data/Modelo_DenseNetFine.h5'

    if os.path.exists(model_path):
        print("Cargando el modelo guardado.")
        model = load_model(model_path)
    else:
        print("Creando un nuevo modelo.")
        model = build_model(20)

    df_filtering = add_new_images_to_df(df_filtering, start_number, num_new_images, '1')
    print("Labels value count :")
    print(df_filtering['labels'].value_counts())
    train_generator,validation_generator = preprocces_images(df_filtering)
    model = build_model()

    start_time = time.time()
    history = train_model(model, train_generator, 55,100, validation_data=validation_generator)
    end_time = time.time()

    plot_training_history(history)
    y_score, y_test = get_predictions_and_labels(model, validation_generator)
    cnf_matrix = confusion_matrix(y_test, y_score)
    plot_confusion_matrix(cnf_matrix, classes=['Normal', 'Diabetes'], normalize=True, title='Normalized confusion matrix')
    print(f"Total training time: {end_time - start_time} seconds")