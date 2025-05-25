import subprocess
import csv
import tempfile
from pathlib import Path
import sys, importlib
import os

sys.path.append(os.path.abspath('../scripts/xTTS-v2'))
from finetune import finetune as xtts_finetune
from synthesize import synthesize as xtts_synthesize

sys.path.append(os.path.abspath('../scripts/orpheusTTS'))
from train_tuning import finetune as orpheus_finetune
from inference_tuning import synthesize as orpheus_synthesize

'''
Função para realizar finetune de um modelo
Recebe:
    - nome do modelo, para nomear a pasta de output
    - tipo do modelo (xtts, orpheus)
    - numero de épocas de treinamento
    - Taxa de aprendizado
    - Tipo de input
'''
def finetune(model_name, model_to_tuning, duration_to_tuning, learning_to_tuning, inputType):

    if inputType == "audio":
        input_path = '../data/audio_transcription'
    else:
        input_path = f'../data/audio_transcription/{model_name}'
    output_path = f'../data/modelos/{model_name}'

    if model_to_tuning == "xTTS-v2":
        print("Entrou no xTTS-v2")
        result = xtts_finetune(input_path, output_path, duration_to_tuning, learning_to_tuning)
    elif model_to_tuning == "orpheusTTS":
        print("Entrou no Orpheus")
        result = orpheus_finetune(input_path, output_path, duration_to_tuning, learning_to_tuning, inputType)

    with open("../data/models.csv", "a") as csv_modelos:
        csv_writer = csv.writer(csv_modelos, delimiter=',')
        csv_writer.writerow([model_name, output_path, model_to_tuning, 0.0])

    print(result)
    return result

'''
Função para sintetizar um áudio usando um dos modelos listados
Recebe:
    - Texto do áudio desejado
    - Path da pasta contendo o modelo
    - Tipo do modelo (xTTs, Orpheus)
'''
def synthesize(text: str, model_path: str, model_type: str):
    input_path = model_path
    output_path = '../data/gen'
    audio_path = ""

    if model_type == "xTTS-v2":
        xtts_synthesize(input_path, output_path, text)
        audio_path = f"{output_path}/output.wav"
    elif model_type == "orpheusTTS":
        orpheus_synthesize(input_path, output_path, text)
        audio_path = f"{output_path}/OutputTTSOrpheus.wav"

    return audio_path

def evaluate_audio_metrics(
    audio_path: str | Path,
    reference_text: str,
    sample_audio_path: str | Path | None = None,
    *,
    lang: str = "pt",
) -> dict[str, float | str]:
    """
    Returns a dict with UTMOS, CER and SECS
    (SECS is "N/A" if no reference sample is given).
    """

    ## Go from /app/ to project root
    project_root = Path(__file__).resolve().parents[1]

    # Path to the folder containing metrics.py
    metrics_folder = project_root / "scripts" / "metrics"

    if str(metrics_folder) not in sys.path:
        sys.path.append(str(metrics_folder))

    # Load the metrics module
    metrics = importlib.import_module("metrics")

    utmos_score = float(metrics.UTMOS(audio_path))
    cer_score = float(metrics.CER(audio_path, reference_text, lang=lang))

    secs_score = (
        float(metrics.SECS(sample_audio_path, audio_path))
        if sample_audio_path else "N/A"
    )

    return {
        "UTMOS (Naturality)"  : round(utmos_score, 3),
        "CER (Pronunciation)" : round(cer_score, 3),
        "SECS (Similarity)"   : secs_score if secs_score == "N/A" else round(secs_score, 3),
    }