import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv("../Data/Eval/NON_NAT.csv")
    check_for = ["O", "B", "U", "J", "X", "-", "r", "h", "x"]

    non_nats = []
    for idx, row in df.iterrows():
        seq = row["Sequence"]
        for nn_char in check_for:
            if seq not in non_nats and nn_char in seq and "[UNK]" not in seq:
                non_nats.append(seq)
    non_nats = pd.DataFrame(non_nats, columns=[["Sequence"]])
    non_nats.to_csv("../Data/Eval/NON_NAT_cleaned.csv")


    
