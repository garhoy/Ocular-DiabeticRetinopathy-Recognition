import pandas as pd





############################################ READING AND EXTRACTING CHARACTERISTICS ################################################

def Reading_csv():
    csv = pd.read_csv("full_df.csv")
    print(csv.head())
    









if __name__ == "__main__":
    Reading_csv()