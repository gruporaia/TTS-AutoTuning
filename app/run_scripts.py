import subprocess
import csv
import tempfile
from pathlib import Path
import sys, importlib
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

    result = subprocess.run(
        [f"../scripts/{model_to_tuning}/finetune.sh", input_path, output_path, duration_to_tuning, learning_to_tuning, inputType],
        check=True,
        capture_output=True,
        text=True
    )

    with open("../data/models.csv", "a") as csv_modelos:
        csv_writer = csv.writer(csv_modelos, delimiter=',')
        csv_writer.writerow([model_name, output_path, model_to_tuning, 0.0])

    print(result)
    return result

#### função de avaliação do modelo após finetune
# só um placeholder até script the avaliação for adicionado
def model_eval(model_path: str):
    sample_text = "Testando ajuste fino de modelo. O rato roeu a roupa do rei de roma. Testando 1 2 3"

    # gera áudio com modelo para ser avaliado
    synthesize(sample_text, model_path)
    output_path = '../data/gen'

    # chama função de avaliação

    '''
    result = subprocess.run(
        [f"Endereço do .sh de avaliação", sample_text, output_path],
        check=True,
        capture_output=True,
        text=True
    )
    '''

    return

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

    result = subprocess.run(
        [f"../scripts/{model_type}/synthesize.sh", input_path, output_path, text],
        check=True,
        capture_output=True,
        text=True
    )

    print(result)

    audio_path = ""

    if model_type == "xTTS-v2":
        audio_path = f"{output_path}/output.wav"
    else:
        audio_path = f"{output_path}/OutputTTSOrpheus.wav"

    return audio_path


'''
Função para gerar áudio utilizado modelo base do Orpheus
Recebe:
    - Texto do áudio desejado
    - Trancrição da amostra de áudio
    - Áudio de referência
'''
def inference_pretrained(text, transcript, audio_sample_path):
    output_path = '../data/gen'

    result = subprocess.run(
        ["../scripts/orpheusTTS/pre_trained.sh", audio_sample_path, output_path, text, transcript],
        check=True,
        capture_output=False,
        text=True
    )

    audio_path = f"{output_path}/outputPretrainedOrpheus.wav"
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