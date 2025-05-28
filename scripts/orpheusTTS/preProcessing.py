# Faz o processamento do dataset de formato (metadata.csv e wavs) em um
# dataset em formato Hugging Face

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
import pandas as pd
from datasets import Dataset, Audio, concatenate_datasets
import glob


def create_dataset(base_dir):
    csv_path = glob.glob(os.path.join(base_dir, "*.csv"))[1]
    audio_dir = os.path.join(base_dir, "wavs")
    print("diretorio base: " + base_dir + " " + csv_path + " " + audio_dir)

    df = pd.read_csv(csv_path, sep="|")
    df["audio"] = df["ID"].apply(lambda x: os.path.join(audio_dir, x + ".wav"))

    df = df.rename(columns={"text": "aux"})
    df = df.rename(columns={"textCleaned": "text"})

    df = df[["audio", "text"]] # Dataset apenas com campos de áudio e texto

    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column("audio", Audio())
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    print(dataset[0])

    return dataset