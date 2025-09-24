import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_split_counts(p12_dir):
    p12_train = pd.read_csv(os.path.join(p12_dir, "physionet_2012_train.csv"))
    p12_val   = pd.read_csv(os.path.join(p12_dir, "physionet_2012_val.csv"))
    n_train = p12_train["pid"].nunique()
    n_val   = p12_val["pid"].nunique()
    return n_train, n_val

def stratified_subsample_by_pid(df, n_target, seed=42):
    # Patient-level labels for stratification
    pid_label = df[["pid", "target"]].drop_duplicates(subset=["pid"])
    pids = pid_label["pid"].values
    y    = pid_label["target"].values.astype(int)

    if len(pids) <= n_target:
        return df  # nothing to subsample

    # Stratified sampling on patient IDs
    sel_idx, _ = train_test_split(
        np.arange(len(pids)),
        train_size=n_target,
        stratify=y,
        random_state=seed,
        shuffle=True
    )
    sel_pids = set(pids[sel_idx])
    return df[df["pid"].isin(sel_pids)].copy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mi3_dir", default="./data/MI3_st/renew", type=str)
    parser.add_argument("--p12_dir", default="./data/P12", type=str)
    parser.add_argument("--out_dir", default="./data/MI3_p12sized", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--keep_test", action="store_true", help="Copy test split without subsampling")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Target counts from P12
    n_train, n_val = load_split_counts(args.p12_dir)
    print(f"P12 unique patients -> train: {n_train}, val: {n_val}")

    # Load MI3 splits
    mi3_train = pd.read_csv(os.path.join(args.mi3_dir, "mimic3_mortality_train.csv"))
    mi3_val   = pd.read_csv(os.path.join(args.mi3_dir, "mimic3_mortality_val.csv"))

    # Subsample
    mi3_train_sub = stratified_subsample_by_pid(mi3_train, n_train, seed=args.seed)
    mi3_val_sub   = stratified_subsample_by_pid(mi3_val, n_val, seed=args.seed)

    # Save
    train_out = os.path.join(args.out_dir, "mimic3_mortality_train.csv")
    val_out   = os.path.join(args.out_dir, "mimic3_mortality_val.csv")
    mi3_train_sub.to_csv(train_out, index=False)
    mi3_val_sub.to_csv(val_out, index=False)
    print(f"Saved: {train_out} (patients={mi3_train_sub['pid'].nunique()} rows={len(mi3_train_sub)})")
    print(f"Saved: {val_out}   (patients={mi3_val_sub['pid'].nunique()} rows={len(mi3_val_sub)})")

    # Optionally copy test as-is
    if args.keep_test:
        src = os.path.join(args.mi3_dir, "mimic3_mortality_test.csv")
        dst = os.path.join(args.out_dir, "mimic3_mortality_test.csv")
        if os.path.exists(src):
            pd.read_csv(src).to_csv(dst, index=False)
            print(f"Copied test split to: {dst}")

if __name__ == "__main__":
    main()