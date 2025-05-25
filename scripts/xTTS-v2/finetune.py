import os
import sys
import shutil
import gc
import time

from pathlib import Path
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.utils.manage import ModelManager
from trainer import Trainer, TrainerArgs

def finetune(dataset_path: str, output_path: str, epochs: int, lr:int):

    print("ENTREIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")

    # Logging parameters
    RUN_NAME = "XTTSV2"
    PROJECT_NAME = "VOICESYNTH"
    DASHBOARD_LOGGER = "tensorboard"
    LOGGER_URI = None
    EVAL_SPLIT_SIZE = 0.2
    EVAL_SPLIT_MAXSIZE = 256

    # Set here the path that the checkpoints will be saved. Default: ./run/training/
    OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run", "training")

    DATASET_PATH = dataset_path

    # Training Parameters
    OPTIMIZER_WD_ONLY_ON_WEIGHTS = True  # for multi-gpu training please make it False
    START_WITH_EVAL = True  # if True it will star with evaluation
    BATCH_SIZE = 3  # set here the batch size
    GRAD_ACUMM_STEPS = 84  # set here the grad accumulation steps
    EPOCHS=70
    # Note: we recommend that BATCH_SIZE * GRAD_ACUMM_STEPS need to be at least 252 for more efficient training. You can increase/decrease BATCH_SIZE but then set GRAD_ACUMM_STEPS accordingly.

    # Define here the dataset that you want to use for the fine-tuning on.
    config_dataset = BaseDatasetConfig(
        formatter="ljspeech",
        dataset_name="bombrilSpeech",
        path=DATASET_PATH,
        meta_file_train="metadata.csv",
        language="pt",
    )

    # Add here the configs of the datasets
    DATASETS_CONFIG_LIST = [config_dataset]

    # Define the path where XTTS v2.0.1 files will be downloaded
    #CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
    #os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)
    CHECKPOINTS_OUT_PATH = "../data/models/XTTS_v2.0_original_model_files"


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

    # load training samples
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=EVAL_SPLIT_MAXSIZE, # botar esse dois como hardcodados e mover pra cima
        eval_split_size=EVAL_SPLIT_SIZE,
    )

    
    # Training sentences generations
    SPEAKER_REFERENCE = [
        train_samples[0]['audio_file']  # speaker reference to be used in training test sentences
    ]
    LANGUAGE = config_dataset.language
    SPEAKER_TEXT = train_samples[0]['text']
    train_samples.pop()



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
    # training parameters config
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
        eval_split_max_size=EVAL_SPLIT_MAXSIZE,
        print_step=50,
        plot_step=100,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={
            "milestones": [90000000, 60, 65], # MultiStepLR atualiza por época, não step. A receita original ta errada
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

    # init the model from config
    model = GPTTrainer.init_from_config(config)

    

    print("TRAIN LENGTH: ", len(train_samples), "\n\n\n\n")
    print(train_samples[0])
    print("EVAL LENGTH: ", len(eval_samples), "\n\n\n\n")
    print(eval_samples[0])

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,  # xtts checkpoint is restored via xtts_checkpoint key so no need of restore it using Trainer restore_path parameter
            skip_train_epoch=False,
            gpu=0,
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

    path = Path(OUT_PATH)
    folders = [f for f in path.iterdir() if f.is_dir()]
    most_recent = max(folders, key=lambda f: f.stat().st_ctime) # meio tosco, olha qual é a pasta mais recentemente criada
    # não achei um jeito de mudar o nome de pasta que o Trainer gera automáticamente

    #final_output_dir = os.path.join(os.getcwd(), "data", output_path)
    if not os.path.isabs(output_path):
        output_path = os.path.join(os.getcwd(), output_path)

    os.makedirs(output_path, exist_ok=True)

    script_output_path = os.path.join(OUT_PATH, most_recent.name)
    model_path = None
    config_path = None
    log_path = None
    sample_path = SPEAKER_REFERENCE[0]

    for filename in os.listdir(script_output_path):
        if filename.endswith('.pth'):
            model_path =  os.path.join(script_output_path, filename)
        elif filename.endswith('.json'):
            config_path = os.path.join(script_output_path, filename)
        elif filename.endswith('.txt'):
            log_path = os.path.join(script_output_path, filename)

    new_model_path = os.path.join(output_path, "model.pth")
    new_config_path = os.path.join(output_path, "config.json")
    new_sample_path = os.path.join(output_path, "sample.wav")
    new_log_path = os.path.join(output_path, "log.txt")
    new_sample_text_path = os.path.join(output_path, "sampletext.txt")

    if os.path.exists(new_model_path):
        os.remove(new_model_path)

    if os.path.exists(new_config_path):
        os.remove(new_config_path)
    
    if os.path.exists(new_sample_path):
        os.remove(new_sample_path)

    if os.path.exists(new_log_path):
        os.remove(new_sample_path)

    with open(new_sample_path, "w") as f:
        f.write(SPEAKER_TEXT)
    

    shutil.move(model_path, new_model_path)
    shutil.move(config_path, new_config_path)
    shutil.copy(sample_path, new_sample_path)
    shutil.move(log_path, new_log_path)

    # tem que testar
    with open(new_config_path, "r") as f:
        config_data = json.load(f)

    config_data["model_args"]["xtts_checkpoint"] = new_model_path

    with open(new_config_path, "w") as f:
        json.dump(config_data, f, indent=4)

    shutil.rmtree(script_output_path, ignore_errors=True)

    return OUT_PATH + most_recent.name