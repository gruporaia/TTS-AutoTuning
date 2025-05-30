import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  

import argparse
import torch
import locale
import torchaudio.transforms as T
import torch
import sys
import shutil
import gc
import time
import preProcessing as pre

from snac import SNAC
from transformers import TrainingArguments,Trainer,DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported, FastLanguageModel

from IPython.display import display, Audio
import torchaudio

def synthesize(text: str, model_path: str, audio_output_path: str):
	# Ensure output directory exists
	os.makedirs(audio_output_path, exist_ok=True)

	snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
	model, tokenizer = FastLanguageModel.from_pretrained(
		model_name = model_path,
		max_seq_length= 4096, # Choose any for long context!
		dtype = torch.float16,
		load_in_4bit = False,
		#token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
	)

	prompts = [text]

	chosen_voice = None # None for single-speaker

	#@title Run Inference

	FastLanguageModel.for_inference(model) # Enable native 2x faster inference

	# Moving snac_model cuda to cpu
	snac_model.to("cpu")

	prompts_ = [(f"{chosen_voice}: " + p) if chosen_voice else p for p in prompts]

	all_input_ids = []

	for prompt in prompts_:
		input_ids = tokenizer(prompt, return_tensors="pt").input_ids
		all_input_ids.append(input_ids)

	start_token = torch.tensor([[ 128259]], dtype=torch.int64) # Start of human
	end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64) # End of text, End of human

	all_modified_input_ids = []
	for input_ids in all_input_ids:
		modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1) # SOH SOT Text EOT EOH
		all_modified_input_ids.append(modified_input_ids)

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

	generated_ids = model.generate(
		input_ids=input_ids,
		attention_mask=attention_mask,
		max_new_tokens=1200,
		do_sample=True,
		temperature=0.6,
		top_p=0.95,
		repetition_penalty=1.1,
		num_return_sequences=1,
		eos_token_id=128258,
		use_cache = True
	)
	token_to_find = 128257
	token_to_remove = 128258

	token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

	if len(token_indices[1]) > 0:
		last_occurrence_idx = token_indices[1][-1].item()
		cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
	else:
		cropped_tensor = generated_ids

	mask = cropped_tensor != token_to_remove

	processed_rows = []

	for row in cropped_tensor:
		masked_row = row[row != token_to_remove]
		processed_rows.append(masked_row)

	code_lists = []

	for row in processed_rows:
		row_length = row.size(0)
		new_length = (row_length // 7) * 7
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

		# codes = [c.to("cuda") for c in codes]
		audio_hat = snac_model.decode(codes)
		return audio_hat

	my_samples = []
	for code_list in code_lists:
		samples = redistribute_codes(code_list)
		my_samples.append(samples)

	if len(prompts) != len(my_samples):
		raise Exception("Number of prompts and samples do not match")
	else:
		for i in range(len(my_samples)):
			print(prompts[i])
			samples = my_samples[i].detach().to("cpu").squeeze()
			if samples.ndim == 1:
				samples = samples.unsqueeze(0)

			filename = os.path.join(audio_output_path, "OutputTTSOrpheus.wav")
			torchaudio.save(filename, samples, 24000)

	# Clean up to save RAM
	del my_samples,samples

	return f"{audio_output_path}/OutputTTSOrpheus.wav"

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Script de síntese - OrpheusTTS")
	parser.add_argument("--text", type=str, required=True)

    parser.add_argument("--model_path", type=str, required=True)

    parser.add_argument("--audio_output_path", type=str, required=True)

    args = parser.parse_args()

    synthesize(args.text, args.model_path, args.audio_output_path)