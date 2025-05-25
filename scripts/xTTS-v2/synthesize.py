import os
import sys
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torchaudio

def synthesize(run_path=None, output_path=None, text=None):
    model_path = config_path = sample_path = None
    if run_path:
        for filename in os.listdir(run_path):
            if filename.endswith('model.pth'):
                model_path =  os.path.join(run_path, filename)
            elif filename.endswith('config.json'):
                config_path = os.path.join(run_path, filename)
            elif filename.endswith('.wav'):
                sample_path = os.path.join(run_path, filename)

        if not model_path or not config_path or not sample_path:
            return 
    

    config = XttsConfig()
    config.load_json(config_path)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_path=model_path,
        eval=True,
    )
    model.cuda()

    # no script de finetune tinha que salvar uma amostra do falante no arquivo do modelo
    # também tem que lembrar de apagar a pasta do finetune
    outputs = model.synthesize(
        text,
        config,
        speaker_wav=sample_path,
        gpt_cond_len=3,
        language="pt",
    )

    if not os.path.isabs(output_path):
        output_path = os.path.join(os.getcwd(), output_path)

    os.makedirs(output_path, exist_ok=True)
    
    output_path = os.path.join(output_path, "output.wav")
    if os.path.exists(output_path):
        os.remove(output_path)
    

    torchaudio.save(output_path, torch.tensor(outputs['wav']).unsqueeze(0), 24000)

