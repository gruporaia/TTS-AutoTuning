# Fine-tuning do Orpheus TTS

import torch
import locale
import torchaudio.transforms as T
import os
import torch
import sys
import shutil
import gc
import time
import preProcessing as pre

from snac import SNAC
from transformers import TrainingArguments,Trainer,DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from datasets import load_dataset, load_from_disk
from unsloth import FastLanguageModel



def finetune(audio_path: str, model_output_path: str, duration: int, learning_rate: float, inputType: str):
    # Ensure output directory exists
    os.makedirs(model_output_path, exist_ok=True)

    # Faz o fine-tuning com menos épocas, já que é extremamente demorado para 40, 50, etc
    auxDict = {
        "50": 1,
        "60": 2,
        "70": 3,
        "80": 4
    }

    if inputType == "audio":
        dataset = pre.create_dataset(audio_path)
    else:
        dataset = load_from_disk(audio_path)

    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/orpheus-3b-0.1-ft-unsloth-bnb-4bit",
        max_seq_length= 4096, # Choose any for long context!
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 64,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    # Tokenização das amostras do dataset

    locale.getpreferredencoding = lambda: "UTF-8"
    ds_sample_rate = dataset[0]["audio"]["sampling_rate"]

    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
    snac_model = snac_model.to("cuda")

    def tokenise_audio(waveform):
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        waveform = waveform.to(dtype=torch.float32)
        resample_transform = T.Resample(orig_freq=ds_sample_rate, new_freq=24000)
        waveform = resample_transform(waveform)

        waveform = waveform.unsqueeze(0).to("cuda")

        #generate the codes from snac
        with torch.inference_mode():
            codes = snac_model.encode(waveform)

        all_codes = []
        for i in range(codes[0].shape[1]):
            all_codes.append(codes[0][0][i].item()+128266)
            all_codes.append(codes[1][0][2*i].item()+128266+4096)
            all_codes.append(codes[2][0][4*i].item()+128266+(2*4096))
            all_codes.append(codes[2][0][(4*i)+1].item()+128266+(3*4096))
            all_codes.append(codes[1][0][(2*i)+1].item()+128266+(4*4096))
            all_codes.append(codes[2][0][(4*i)+2].item()+128266+(5*4096))
            all_codes.append(codes[2][0][(4*i)+3].item()+128266+(6*4096))

        return all_codes

    def add_codes(example):
        # Always initialize codes_list to None
        codes_list = None

        try:
            answer_audio = example.get("audio")
            # If there's a valid audio array, tokenise it
            if answer_audio and "array" in answer_audio:
                audio_array = answer_audio["array"]
                codes_list = tokenise_audio(audio_array)
        except Exception as e:
            print(f"Skipping row due to error: {e}")
            # Keep codes_list as None if we fail
        example["codes_list"] = codes_list

        return example

    dataset = dataset.map(add_codes, remove_columns=["audio"])

    tokeniser_length = 128256
    start_of_text = 128000
    end_of_text = 128009

    start_of_speech = tokeniser_length + 1
    end_of_speech = tokeniser_length + 2

    start_of_human = tokeniser_length + 3
    end_of_human = tokeniser_length + 4

    start_of_ai = tokeniser_length + 5
    end_of_ai =  tokeniser_length + 6
    pad_token = tokeniser_length + 7

    audio_tokens_start = tokeniser_length + 10

    dataset = dataset.filter(lambda x: x["codes_list"] is not None)
    dataset = dataset.filter(lambda x: len(x["codes_list"]) > 0)

    def remove_duplicate_frames(example):
        vals = example["codes_list"]
        if len(vals) % 7 != 0:
            raise ValueError("Input list length must be divisible by 7")

        result = vals[:7]

        removed_frames = 0

        for i in range(7, len(vals), 7):
            current_first = vals[i]
            previous_first = result[-7]

            if current_first != previous_first:
                result.extend(vals[i:i+7])
            else:
                removed_frames += 1

        example["codes_list"] = result

        return example

    dataset = dataset.map(remove_duplicate_frames)

    tok_info = '''*** HERE you can modify the text prompt
    If you are training a multi-speaker model (e.g., canopylabs/orpheus-3b-0.1-ft),
    ensure that the dataset includes a "source" field and format the input accordingly:
    - Single-speaker: f"{example['text']}"
    - Multi-speaker: f"{example['source']}: {example['text']}"
    '''

    def create_input_ids(example):
        # Determine whether to include the source field
        text_prompt = f"{example['source']}: {example['text']}" if "source" in example else example["text"]

        text_ids = tokenizer.encode(text_prompt, add_special_tokens=True)
        text_ids.append(end_of_text)

        example["text_tokens"] = text_ids
        input_ids = (
            [start_of_human]
            + example["text_tokens"]
            + [end_of_human]
            + [start_of_ai]
            + [start_of_speech]
            + example["codes_list"]
            + [end_of_speech]
            + [end_of_ai]
        )
        example["input_ids"] = input_ids
        example["labels"] = input_ids
        example["attention_mask"] = [1] * len(input_ids)

        return example

    dataset = dataset.map(create_input_ids, remove_columns=["text", "codes_list"])
    columns_to_keep = ["input_ids", "labels", "attention_mask"]
    columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]

    dataset = dataset.remove_columns(columns_to_remove)

    # Treinamento do modelo
    trainer = Trainer(
        model = model,
        train_dataset = dataset,
        args = TrainingArguments(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 6,
            warmup_steps = 5,
            num_train_epochs = duration,
            max_steps = -1,
            learning_rate = learning_rate,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            save_strategy = "no",
            output_dir = "../../data",
            report_to = "none",
        ),
    )

    trainer_stats = trainer.train()

    # Este modo de salvamento salva o modelo por completo, contando tanto o modelo padrão do Orpheus
    # quanto os adaptadores lora (é mais pesado e lento)
    # model.save_pretrained_merged(model_output_path, tokenizer, save_method = "merged_16bit",)

    model.save_pretrained(model_output_path)
    tokenizer.save_pretrained(model_output_path)

    # Exclui os arquivos cache que são gerados na pasta do dataset
    if inputType == "zip":
        for file_name in os.listdir(audio_path):
            full_path = os.path.join(audio_path, file_name)
        
            if os.path.isfile(full_path) and file_name.startswith("cache"):
                os.remove(full_path)