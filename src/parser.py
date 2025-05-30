import argparse


# Args parser
def parse_args():
    parser = argparse.ArgumentParser(description="CSAT_Kor settings")

    parser.add_argument("--seed", default=42, type=int, help="Reproducibility")
    parser.add_argument("--dataset", default="KoreanData.csv", type=str, help="Dataset")
    parser.add_argument("--model_type", default="roberta", type=str, help="Method")
    parser.add_argument("--batch_size", default=32, type=int, help="Size of Batch")
    parser.add_argument("--epoch", default=10, type=int, help="Range of Epoch")

    args = parser.parse_args()

    return args
