import pandas as pd
from glob import glob
import os

files = glob("data/raw/MetaMotion/*.csv")


# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------
def read_data_from_files(files: list) -> pd.DataFrame:
    """read data from files and return accelerometer and gyroscope dataframes and merge them

    Args:
        files (list): all files path in data/raw/MetaMotion

    Returns:
        pd.DataFrame: _description_
    """
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()
    env_path = r"c:\Users\GTR\Documents\GitHub\Fitness_Tracker\src\data"
    data_path = "data/raw/MetaMotion\\"
    acc_set = 1
    gyr_set = 1

    for f in files:
        if os.getcwd() == env_path:
            participant = f.split("-")[0].replace("../../" + data_path, "")
        else:
            participant = f.split("-")[0].replace(data_path, "")
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019.csv")

        df = pd.read_csv(f)

        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])
        else:
            df["set"] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df])

    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

    acc_df.drop(
        ["epoch (ms)", "time (01:00)", "elapsed (s)"], axis=1, inplace=True
    )  # drop columns
    gyr_df.drop(
        ["epoch (ms)", "time (01:00)", "elapsed (s)"], axis=1, inplace=True
    )  # drop columns

    return acc_df, gyr_df


acc_df, gyr_df = read_data_from_files(files)
print("Done")
# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------

# merging both accelerometer and gyroscope data into one dataframe
data_merged = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)

# renaming columns
data_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set",
]
print("Done")

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------
"""
# Accelerometer sensor:    12.500HZ  1/12.500 = 0.08s 80ms
# Gyroscope sensor:        25.000Hz  1/25.000 = 0.04s 40ms

# resampling convert frequency to 200ms

"""


def resampling(data_merged: pd.DataFrame, rule: str = "200ms") -> pd.DataFrame:
    """_summary_

    Args:
        data_merged (pd.DataFrame): _description_
        rule (str, optional): _description_. Defaults to "200ms".

    Returns:
        pd.DataFrame: _description_
    """

    # resampling aggregation rules
    sampling = {
        "acc_x": "mean",
        "acc_y": "mean",
        "acc_z": "mean",
        "gyr_x": "mean",
        "gyr_y": "mean",
        "gyr_z": "mean",
        "participant": "last",
        "label": "last",
        "category": "last",
        "set": "last",
    }
    # split the data into days and resample each day separately and then concatenate the results
    days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]

    # resample each day separately with the sampling rules and 0.2s frequency
    data_resampled = pd.concat(
        [df.resample(rule=rule).apply(sampling).dropna() for df in days]
    )
    # convert set to int
    data_resampled["set"] = data_resampled["set"].astype(int)

    return data_resampled


data_resampled = resampling(data_merged, "200ms")
print("Done")

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

data_resampled.to_pickle("data/interim/01_data_resampled.pkl")
print("Done")
