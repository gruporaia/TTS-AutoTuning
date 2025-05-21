# Faz o processamento do dataset de formato (metadata.csv e wavs) em um
# dataset em formato Hugging Face

import os
import pandas as pd
from datasets import Dataset, Audio, concatenate_datasets

def create_dataset(base_dir):
    csv_path = os.path.join(base_dir, "metadata.csv")
    audio_dir = os.path.join(base_dir, "wavs")

    df = pd.read_csv(csv_path, sep="|", header=None, names=["audio_filename", "text_original", "text"])
    df["audio"] = df["audio_filename"].apply(lambda x: os.path.join(audio_dir, x + ".wav"))
    df = df[["audio", "text"]] # Dataset apenas com campos de áudio e texto

    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column("audio", Audio())
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    return dataset