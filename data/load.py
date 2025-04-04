import pandas as pd
import re


# Function to pre-process the texts
def preprocess(text):

    # re library
    text = re.sub(r"[󰡔󰡕]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text


# Function to encoding labels
def label_encoding(df, column):

    # Mapping dictionary
    class_mapping = {"최상": 5, "상": 5, "중상": 4, "중": 3, "중하": 2, "하": 1}

    # Mapping
    df[column] = df[column].map(class_mapping)

    return df


# Function to load data
def load_data(csv_file):

    # Load csv file
    korean_df = pd.read_csv(csv_file)

    # Drop unnecessary columns
    korean_df = korean_df.drop(
        columns=[
            "input_text",
            "answer_rate",
            "sentence_length",
            "lexical_diversity",
            "syntactic_complexity",
        ]
    )

    # Text pre-processing
    korean_df["input_text_long"] = korean_df["input_text_long"].map(preprocess)

    # Label encoding
    korean_df = label_encoding(korean_df, "difficulty_label")

    return korean_df


if __name__ == "__main__":
    korean_dataset = load_data("KoreanData.csv")
    print(korean_dataset.loc[0, "input_text_long"])
