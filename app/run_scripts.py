import subprocess
import csv
import tempfile
from pathlib import Path
import sys, importlib
import os
from TTS_Utils import UTMOS, SECS, CER

xtts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'xTTS-v2'))
if xtts_path not in sys.path:
    sys.path.insert(0, xtts_path)

from finetune import finetune as xtts_finetune
from synthesize import synthesize as xtts_synthesize

orpheus_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'orpheusTTS'))
if orpheus_path not in sys.path:
    sys.path.insert(0, orpheus_path)

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
        input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'audio_transcription'))
    else:
        input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'audio_transcription', f'{model_name}'))
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'modelos', f'{model_name}'))

    if model_to_tuning == "xTTS-v2":
        result = xtts_finetune(input_path, output_path, duration_to_tuning, learning_to_tuning)
    elif model_to_tuning == "orpheusTTS":
        result = orpheus_finetune(input_path, output_path, duration_to_tuning, learning_to_tuning, inputType)

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
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'gen'))
    audio_path = ""

    
    if model_type == "xTTS-v2":
        xtts_synthesize(text, output_path, input_path)
        audio_path = f"{output_path}/output.wav"
    elif model_type == "orpheusTTS":
        synthesize_Orpheus_path = os.path.join(orpheus_path, "inference_tuning.py")

        cmd = [
            sys.executable,
            synthesize_Orpheus_path,
            "--text", text,
            "--model_path", input_path,
            "--audio_output_path", output_path
        ]

        subprocess.run(cmd, check=True, capture_output=True, text=True)
        #orpheus_synthesize(text, input_path, output_path)
        audio_path = f"{output_path}/OutputTTSOrpheus.wav"

    return audio_path

def evaluate_audio_metrics(
   audio_path: str | Path,
    reference_text: str,
    sample_audio_path: str | Path | None = None,
    *,
    lang: str = "pt"
) -> dict[str, float | str]:
    """
    Returns a dict with UTMOS, CER and SECS
    (SECS is "N/A" if no reference sample is given).

    ''' ,''''
    """


    utmos_score = float(UTMOS(audio_path))
    cer_score = float(CER(audio_path, reference_text, lang=lang))

    secs_score = (
        float(SECS(sample_audio_path, audio_path))
        if sample_audio_path else "N/A"
    )

    return {
        "UTMOS (Naturality)"  : round(utmos_score, 3),
        "CER (Pronunciation)" : round(cer_score, 3),
        "SECS (Similarity)"   : secs_score if secs_score == "N/A" else round(secs_score, 3),
    }
