# Síntese de áudio com Orpheus TTS pré-treinado e apenas
# 1 amostra de áudio

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  

import torch
import numpy as np
import soundfile as sf
import librosa
import torchaudio.transforms as T
import torch
import torchaudio
import sys
import shutil
import gc
import time

from datasets import load_dataset, load_from_disk
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
from unsloth import is_bfloat16_supported, FastLanguageModel
from huggingface_hub import snapshot_download
from snac import SNAC

def pre_trained(input_path: str, output_path: str, text: str, transcript: str):
	# Ensure output directory exists
	os.makedirs(output_path, exist_ok=True)
   
	model_name = "canopylabs/orpheus-tts-0.1-pretrained"
	tokenizer = AutoTokenizer.from_pretrained(model_name)

	device = "cuda" if torch.cuda.is_available() else "cpu"
	snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
   
	model_path = snapshot_download(
		repo_id=model_name,
		allow_patterns=[
			"config.json",
			"*.safetensors",
			"model.safetensors.index.json",
		],
		ignore_patterns=[
			"optimizer.pt",
			"pytorch_model.bin",
			"training_args.bin",
			"scheduler.pt",
			"tokenizer.json",
			"tokenizer_config.json",
			"special_tokens_map.json",
			"vocab.json",
			"merges.txt",
			"tokenizer.*"
		]
	)
	
	model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
	model.cuda()

	my_wav_file_is = input_path
	and_the_transcript_is = transcript
	the_model_should_say = [text]

	''' Here we tokenise the prompt you gave us, we also tokenise the prompts you want the model to say

	The template is:

	start_of_human, start_of_text, text, end_of_text, start_of_ai, start_of_speech, speech, end_of_speech, end_of_ai, start_of_human, text, end_of_human and then generate from here

	'''

	# Tokenização do que o modelo deve falar e da transcrição do áudio de input

	filename = my_wav_file_is

	audio_array, sample_rate = librosa.load(filename, sr=24000)

	def tokenise_audio(waveform):
		waveform = torch.from_numpy(waveform).unsqueeze(0)
		waveform = waveform.to(dtype=torch.float32)
		waveform = waveform.unsqueeze(0)

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

	myts = tokenise_audio(audio_array)
	start_tokens = torch.tensor([[ 128259]], dtype=torch.int64)
	end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
	final_tokens = torch.tensor([[128258, 128262]], dtype=torch.int64)
	voice_prompt = and_the_transcript_is
	prompt_tokked = tokenizer(voice_prompt, return_tensors="pt")

	input_ids = prompt_tokked["input_ids"]

	zeroprompt_input_ids = torch.cat([start_tokens, input_ids, end_tokens, torch.tensor([myts]), final_tokens], dim=1) # SOH SOT Text EOT EOH

	prompts = the_model_should_say

	all_modified_input_ids = []
	for prompt in prompts:
		input_ids = tokenizer(prompt, return_tensors="pt").input_ids
		second_input_ids = torch.cat([zeroprompt_input_ids, start_tokens, input_ids, end_tokens], dim=1)
		all_modified_input_ids.append(second_input_ids)


	all_padded_tensors = []
	all_attention_masks = []

	max_length = max([modified_input_ids.shape[1] for modified_input_ids in all_modified_input_ids])

	for modified_input_ids in all_modified_input_ids:
		padding = max_length - modified_input_ids.shape[1]
		padded_tensor = torch.cat([torch.full((1, padding), 128263, dtype=torch.int64), modified_input_ids], dim=1)
		attention_mask = torch.cat([torch.zeros((1, padding), dtype=torch.int64), torch.ones((1, modified_input_ids.shape[1]), dtype=torch.int64)], dim=1)
		all_padded_tensors.append(padded_tensor)
		all_attention_masks.append(attention_mask)

	all_padded_tensors = torch.cat(all_padded_tensors, dim=0)
	all_attention_masks = torch.cat(all_attention_masks, dim=0)

	input_ids = all_padded_tensors.to("cuda")
	attention_mask = all_attention_masks.to("cuda")

	# Parâmetros funcionam como em uma LLM
	with torch.no_grad():
		generated_ids = model.generate(
			input_ids=input_ids,
			max_new_tokens=8192,
			do_sample=True,
			temperature=0.55,
			top_p=0.95,
			repetition_penalty=1.1,
			num_return_sequences=1,
			eos_token_id=128258,
		)


	# Gera áudio

	token_to_find = 128257
	token_to_remove = 128258

	# Check if the token exists in the tensor
	token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

	if len(token_indices[1]) > 0:
		last_occurrence_idx = token_indices[1][-1].item()
		cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
	else:
		cropped_tensor = generated_ids

	mask = cropped_tensor != token_to_remove
	processed_rows = []
	for row in cropped_tensor:
		# Apply the mask to each row
		masked_row = row[row != token_to_remove]
		processed_rows.append(masked_row)

	code_lists = []
	for row in processed_rows:
		# row is a 1D tensor with its own length
		row_length = row.size(0)
		new_length = (row_length // 7) * 7  # largest multiple of 7 that fits in this row
		trimmed_row = row[:new_length]
		trimmed_row = [t - 128266 for t in trimmed_row]
		code_lists.append(trimmed_row)

	def redistribute_codes(code_list):
		layer_1 = []
		layer_2 = []
		layer_3 = []
		for i in range((len(code_list)+1)//7):
			layer_1.append(code_list[7*i])
			layer_2.append(code_list[7*i+1]-4096)
			layer_3.append(code_list[7*i+2]-(2*4096))
			layer_3.append(code_list[7*i+3]-(3*4096))
			layer_2.append(code_list[7*i+4]-(4*4096))
			layer_3.append(code_list[7*i+5]-(5*4096))
			layer_3.append(code_list[7*i+6]-(6*4096))
		codes = [torch.tensor(layer_1).unsqueeze(0),
			torch.tensor(layer_2).unsqueeze(0),
			torch.tensor(layer_3).unsqueeze(0)]
		audio_hat = snac_model.decode(codes)
		return audio_hat

	# Salva áudio gerado
	my_samples = []
	my_samples2 = []
	for code_list in code_lists:
		samples = redistribute_codes(code_list)
		my_samples.append(samples)

	for i in range(len(my_samples)):
		samples = my_samples[i].detach().to("cpu").squeeze()
		if samples.ndim == 1:
			samples = samples.unsqueeze(0)

		filename = os.path.join(output_path, "outputPretrainedOrpheus.wav")
		torchaudio.save(filename, samples, 24000)
	
	audio_path = f"{output_path}/outputPretrainedOrpheus.wav"
	return audio_path