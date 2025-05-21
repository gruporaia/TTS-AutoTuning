import os
import tempfile
import atexit
import shutil
import soundfile as sf
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split


from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.utils.manage import ModelManager


# Logging parameters
RUN_NAME = "XTTSV2"
PROJECT_NAME = "VOICESYNTH"
DASHBOARD_LOGGER = "tensorboard"
LOGGER_URI = None

# Set here the path that the checkpoints will be saved. Default: ./run/training/
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run", "training")


# Training Parameters
OPTIMIZER_WD_ONLY_ON_WEIGHTS = True  # for multi-gpu training please make it False
START_WITH_EVAL = True  # if True it will star with evaluation
BATCH_SIZE = 3  # set here the batch size
GRAD_ACUMM_STEPS = 84  # set here the grad accumulation steps
EPOCHS=25
# Note: we recommend that BATCH_SIZE * GRAD_ACUMM_STEPS need to be at least 252 for more efficient training. You can increase/decrease BATCH_SIZE but then set GRAD_ACUMM_STEPS accordingly.

TEMP_AUDIO_DIR = tempfile.mkdtemp(prefix="xtts_audio_")
atexit.register(lambda: shutil.rmtree(TEMP_AUDIO_DIR, ignore_errors=True))

def prepare_samples(hf_dataset, max_samples=None):
    samples = []
    for idx, item in enumerate(hf_dataset):
        if max_samples and idx >= max_samples:
            break

        if item['speaker_id'] != '12249':
            continue

        audio_path = os.path.join(TEMP_AUDIO_DIR, f"{item['id'].replace('/', '_')}.flac")
        sf.write(audio_path, item["audio"]["array"],
                 item["audio"]["sampling_rate"], subtype='PCM_24')

        samples.append({
            'text': item["transcript"].strip(),
            'audio_file': audio_path,
            'speaker_name': f"MLS_{item['speaker_id']}",
            'root_path': TEMP_AUDIO_DIR,
            'language': 'pt',
            'audio_unique_name': f"mls_portuguese#{item['id'].replace('/', '_')}"
        })

    return samples


# Define the path where XTTS v2.0.1 files will be downloaded
CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)


# DVAE files
DVAE_CHECKPOINT_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/dvae.pth"
MEL_NORM_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/mel_stats.pth"

# Set the path to the downloaded files
DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))

# download DVAE files if needed
if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
    print(" > Downloading DVAE files!")
    ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)


# Download XTTS v2.0 checkpoint if needed
TOKENIZER_FILE_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/vocab.json"
XTTS_CHECKPOINT_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/model.pth"

# XTTS transfer learning parameters: You we need to provide the paths of XTTS model checkpoint that you want to do the fine tuning.
TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))  # vocab.json file
XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))  # model.pth file

# download XTTS v2.0 files if needed
if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
    print(" > Downloading XTTS v2.0 files!")
    ModelManager._download_model_files(
        [TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
    )

# init args and config
model_args = GPTArgs(
    max_conditioning_length=132300,  # 6 secs
    min_conditioning_length=66150,  # 3 secs
    debug_loading_failures=False,
    max_wav_length=255995,  # ~11.6 seconds
    max_text_length=200,
    mel_norm_file=MEL_NORM_FILE,
    dvae_checkpoint=DVAE_CHECKPOINT,
    xtts_checkpoint=XTTS_CHECKPOINT,  # checkpoint path of the model that you want to fine-tune
    tokenizer_file=TOKENIZER_FILE,
    gpt_num_audio_tokens=1026,
    gpt_start_audio_token=1024,
    gpt_stop_audio_token=1025,
    gpt_use_masking_gt_prompt_approach=True,
    gpt_use_perceiver_resampler=True,
)
# define audio config
audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)

print("Loading and preparing dataset...")
mls_pt = load_dataset("facebook/multilingual_librispeech", "portuguese", split="train", streaming=True)
xtts_samples = prepare_samples(mls_pt, max_samples=1000)  # Remove max_samples for full dataset
print("NUMBER OF SAMPLES: ", len(xtts_samples))

# Split samples
train_samples, eval_samples = train_test_split(xtts_samples, test_size=0.1, random_state=42)
print("TOTAL TRAIN SAMPLES: ", len(train_samples))

SPEAKER_REFERENCE = [train_samples[0]["audio_file"]] if train_samples else []
SPEAKER_TEXT = train_samples[0]["text"]if train_samples else ""

config = GPTTrainerConfig(output_path=OUT_PATH,
        epochs=EPOCHS,
        save_checkpoints=False, # tudo isso é só pra não ficar salvando checkpoints do modelo
        save_on_interrupt = True,
        log_model_step = None,
        save_step = 999999999, # ignora
        save_best_after=999999999, # ignora
        model_args=model_args,
        run_name=RUN_NAME,
        project_name="XTTS_trainer",
        run_description="XTTS v2 fine-tuning on MLS Portuguese",
        dashboard_logger="tensorboard",
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=48,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=8,
        eval_split_max_size=256,
        print_step=50,
        plot_step=100,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={
            "milestones": [50, 60, 70], # MultiStepLR atualiza por época, não step. A receita original ta errada
            "gamma": 0.5,
            "last_epoch": -1
        },
        test_sentences=[
            {
                "text": SPEAKER_TEXT,
                "speaker_wav": SPEAKER_REFERENCE,
                "language": "pt",
            },
        ],
    )

# Initialize and train
model = GPTTrainer.init_from_config(config)
trainer = Trainer(
    TrainerArgs(
        restore_path=None,
        gpu=0,
        skip_train_epoch=False,
        start_with_eval=START_WITH_EVAL,
        grad_accum_steps=GRAD_ACUMM_STEPS,
    ),
    config,
    output_path=OUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()
trainer.save_checkpoint()