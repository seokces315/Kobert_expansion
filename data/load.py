import pandas as pd
import re


# Function to pre-process the texts
def preprocess(text):

    # re library
    text = re.sub(r"[󰡔󰡕]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text


# Function to encoding labels
def label_encoding(label):

    # Branch
    # if label == "하":
    #     label = 0
    # elif label == "중하":
    #     label = 1
    # elif label == "중":
    #     label = 2
    # elif label == "중상":
    #     label = 3
    # else:
    #     label = 4

    # Mapping
    label_map = {"최상": 4, "상": 4, "중상": 3, "중": 2, "중하": 1, "하": 0}

    return label_map[label]


# Function to load data
def load_data(csv_file):

    # Load csv file
    korean_data_df = pd.read_csv(csv_file)

    # Select necessary columns
    korean_data_df = korean_data_df[["input_text_long", "difficulty_label"]]

    # Text pre-processing
    korean_data_df["text"] = korean_data_df["input_text_long"].map(preprocess)
    korean_data_df.drop("input_text_long", axis=1, inplace=True)

    # Label encoding
    korean_data_df["label"] = korean_data_df["difficulty_label"].map(label_encoding)
    korean_data_df.drop("difficulty_label", axis=1, inplace=True)

    return korean_data_df


if __name__ == "__main__":
    csat_kor_dataset = load_data("KoreanData.csv")
