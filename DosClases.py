import pandas as pd
from tensorflow.keras.preprocessing.image import load_img
import random
import PIL

############################################ READING AND EXTRACTING CHARACTERISTICS ################################################

def Reading_csv():
    df = pd.read_csv("full_df.csv")
    df_filtering = df[(df['N'] == 1) | (df['D'] == 1)]

    filename = df_filtering[["filename"]]
    labels = df_filtering[["labels"]]

    return labels,filename


def Visualize_RandomImage(images):
    random_file_name = random.choice(list(images.keys()))
    random_image_data = images[random_file_name]
    random_image_data.show()


def Reading_images(filenames):
    images = {}

    for file_name in filenames['filename']:
        image_data = load_img("Train/" + file_name)
        images[file_name] = image_data

    #Visualize_RandomImage(images)
    return images


if __name__ == "__main__":
    labels, filename = Reading_csv()
    images = Reading_images(filename)