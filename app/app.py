import streamlit as st
import shutil
import sys
import time
import os
import csv
import zipfile
import pandas as pd
from TTS_Utils import UTMOS, SECS, CER, build_dataset, normalize_text
import run_scripts as run

orpheus_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'orpheusTTS'))
if orpheus_path not in sys.path:
    sys.path.insert(0, orpheus_path)
from pre_trained import pre_trained as orpheus_pre_trained



# ─────────────── page config ───────────────
st.set_page_config(
    page_title="",
    page_icon=":microphone:",
    layout="centered",
)

# ─────────────── estilos ───────────────
st.markdown(
    """
    <style>
    /* fundo da página */
    body {
        background-color: white !important;
    }

    /* estiliza TODO o app como uma caixa */
    [data-testid="stApp"] > div:first-child {
        background-color: rgba(230, 240, 255, 0.8) !important;
        border: 2px solid #1e90ff !important;
        border-radius: 12px !important;
        padding: 2rem !important;
        margin: 2rem auto !important;
        max-width: 700px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
    }

    /* esconde menu, header e footer padrões */
    #MainMenu, header, footer {
        visibility: hidden !important;
    }

    /* título centralizado e cor */
    h1, .stMarkdown h1 {
        color: #FFFFFF !important;
        text-align: center !important;
    }

    /* botões */
    .stButton>button {
        background-color: #1e90ff !important;
        color: white !important;
        border-radius: 6px !important;
        height: 3em !important;
        width: 100% !important;
        font-size: 1rem !important;
    }

    /* inputs e textarea */
    .stTextArea textarea,
    .stTextInput input {
        border: 2px solid #1e90ff !important;
        border-radius: 6px !important;
    }

    /* barra de progresso */
    [data-testid="stProgress"] div div div {
        background-color: #1e90ff !important;
        height: 10px !important;
        border-radius: 8px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ───────── helper de progresso ─────────
def fake_progress():
    p = st.progress(0)
    for i in range(21):
        time.sleep(0.03)
        p.progress(i / 20)
    p.empty()

# ─────────────── UI ───────────────
###### Fine-tuning
st.title("Finetuning Automatizado & Geração de Fala")
st.divider()

st.subheader("Treine seu Modelo")
st.write(
    "Envie arquivos de áudio para treinar um modelo de fala personalizado. "
    "O modelo será treinado para reproduzir fielmente as vozes informadas e você poderá gerar áudio com ele. "
    "Recomendamos que escolha áudios com boa qualidade e sem ruídos, isolando as vozes desejadas." 
    )
    
uploaded_files = st.file_uploader(
    "Envie arquivos de áudio para ajuste fino em formato WAV, MP3 ou ZIP (com vários arquivos dentro).",
    type=["wav", "mp3", ".zip"],
    accept_multiple_files=True,
    key="fine_tune_files",
)

avaliables_models = {
    "Modelo xTTS-v2 (PT-BR)": "xTTS-v2",
    "Modelo OrpheusTTS (EN)": "orpheusTTS"
}

avaliables_durations = {
    "1": "1",
    "50": "50",
    "60": "60",
    "70": "70",
    "80": "80"
}

avaliables_learning = {
    "2e-4": "0.0002",
    "3e-4": "0.0003",
    "2e-5": "0.00002",
    "3e-5": "0.00003"
}

speaker_name = st.text_area(
    "Digite o nome do modelo:",
    placeholder="Ex.: Sotaque_Carioca",
)

model_to_tuning = st.radio("Selecione o modelo para fine-tuning", avaliables_models.keys())
duration_to_tuning = st.radio("Selecione o número de épocas/steps para fine-tuning", avaliables_durations.keys())
learning_to_tuning = st.radio("Selecione a taxa de aprendizado", avaliables_learning.keys())

exemple_voice_path = None                              

if st.button("Iniciar Ajuste Fino", key="fine_tune"):
    if uploaded_files:
        input_audio_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_audio'))
        os.makedirs(input_audio_path, exist_ok=True)

        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            file_path = os.path.join(input_audio_path, filename)

            if filename.endswith('.zip'):
                inputType = "zip"
                # Cria um caminho para a pasta com o mesmo nome do zip (sem extensão)
                unzip_folder_name = speaker_name
                unzip_folder_path = os.path.join(input_audio_path, unzip_folder_name)
                os.makedirs(unzip_folder_path, exist_ok=True)

                # Salva o zip temporariamente
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())

                # Descompacta
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(unzip_folder_path)

                # Remove o zip depois, se quiser
                os.remove(file_path)
                st.success(f"Arquivo {filename} descompactado em {unzip_folder_path}")

            else:
                inputType = "audio"
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())

            if exemple_voice_path is None:                      # <<< guarda o 1º arquivo
                exemple_voice_path = file_path
    
        with st.spinner("Processando áudio…"):
            input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_audio'))
            output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'audio_transcription'))
            build_dataset(input_dir, output_dir)
            st.success("Áudio processado")
            

            # salva um audio de exemplo para as métricas
            first_generated_wav = None
            for root, _, files in os.walk(output_dir):
                for f in sorted(files):
                    if f.lower().endswith(".wav"):
                        first_generated_wav = os.path.join(root, f)
                        break
                if first_generated_wav:
                    break

            if first_generated_wav:
                fine_tune_samples_dir = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), '..', 'data', 'fine_tune_samples')
                )
                os.makedirs(fine_tune_samples_dir, exist_ok=True)

                saved_sample_path = os.path.join(
                    fine_tune_samples_dir, f"{speaker_name}_sample.wav"
                )
            shutil.copy(first_generated_wav, saved_sample_path)
            st.session_state["fine_tune_sample_path"] = saved_sample_path

            if os.path.exists(input_dir) and os.path.isdir(input_dir):
                shutil.rmtree(input_dir)  

        with st.spinner("Fazendo ajuste fino no modelo"):
            print(run.finetune(speaker_name, avaliables_models[model_to_tuning], (avaliables_durations[duration_to_tuning]), (avaliables_learning[learning_to_tuning]), inputType))
            st.success("Ajuste fino concluído!")

            #apagar a pasta de dataset criada
            output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'audio_transcription'))
            if os.path.exists(output_dir) and os.path.isdir(output_dir):
                shutil.rmtree(output_dir)
                     

    else:
        st.warning("Por favor, envie arquivos de áudio antes de iniciar o ajuste fino.")

st.divider()

###### Inferência
st.subheader("Gerar Áudio")

st.write("Selecione um modelo treinado e insira o texto para gerar o áudio correspondente. "
    "Caso seu modelo personalizado não esteja aparecendo entre as opções, recarregue a página."
    )

text_input = st.text_area(
    "Digite o texto para gerar o áudio:",
    placeholder="Ex.: Olá, seja bem-vindo…",
    key="text_to_generate",
)

models = {}
models_type = {} # Se o modelo é o xtts ou orpheus
#model_evals = {}

pasta_modelos = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'modelos'))

for nome in os.listdir(pasta_modelos):
    path = os.path.join(pasta_modelos, nome)
    if not os.path.isdir(path):
        continue
    files = os.listdir(path)
    if any(f.endswith(".pth") for f in files):
        tipo = "xTTS-v2"
    else:
        tipo = "orpheusTTS"
    
    models[nome] = path
    models_type[nome] = tipo


###### Inferência de modelos
model_select = st.radio("Selecione modelo", models.keys())

sample_audio_path = ""
normalized_transcript = ""

# Orpheus pré-treinado
if model_select == "OrpheusTTS":
    orpheus_audio_sample = st.file_uploader(
        "Envie uma amostra de áudio de até 15s para o OrpheusTTS:",
        type=["wav", "mp3"],
        key="orpheus_audio_sample",
    )

    transcript = st.text_area(
        "Digite o texto falado no áudio:",
        placeholder="Ex.: Olá, seja bem-vindo…",
        key="transcript",
    )

    if orpheus_audio_sample:
        input_audio_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_audio'))

        os.makedirs(input_audio_path, exist_ok=True)

        st.warning("Arquivo Recebido!")

        sample_audio_path = os.path.join(input_audio_path, "sample.wav")
        with open(sample_audio_path, "wb") as f:
            f.write(orpheus_audio_sample.read())

        normalized_transcript = normalize_text(transcript)

# xTTS-v2
if model_select == "XTTS_v2.0_original_model_files":
    # se o modelo selecionado for o xTTs-v2 base, pede uma amostra de áudio referência para o usuário
    xtts_audio_sample = st.file_uploader(
        "Envie uma amostra de áudio para o modelo pré treinado AQUI:",
        type=["wav", "mp3"],
        key="xtts_audio_sample",
    )
    
    if(xtts_audio_sample):
        sample_audio_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_audio'))
        os.makedirs(sample_audio_dir, exist_ok=True) 
        sample_audio_path = os.path.join(sample_audio_dir, xtts_audio_sample.name)
        sample_byte = xtts_audio_sample.getvalue()
        with open(sample_audio_path, "wb") as f:
            f.write(sample_byte)

if st.button("Gerar Áudio", key="generate_audio"):
    audio_path = ""
    if text_input.strip():
        with st.spinner("Gerando áudio…"):
            print(f'model: {model_select}')
            fake_progress()
            normalized_text = text_input

            output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'gen'))

            # Orpheus Pré-treinado
            if model_select == "OrpheusTTS" and sample_audio_path:
                audio_path = orpheus_pre_trained(sample_audio_path, output_path, normalized_text, normalized_transcript)
            # xTTS pré-treinado
            elif model_select == "XTTS_v2.0_original_model_files" and sample_audio_path: # garante que a amostra foi enviada
                audio_path = run.synthesize(normalized_text, models[model_select], models_type[model_select])
            # Demais modelos fine-tunados
            elif model_select != "XTTS_v2.0_original_model_files":
                audio_path = run.synthesize(normalized_text, models[model_select], models_type[model_select])

        if  audio_path:
            st.audio(audio_path)
            with open(audio_path, "rb") as audio_arquivo:
                audio_bytes = audio_arquivo.read()

            is_finetuned_model = model_select not in {
                "XTTS_v2.0_original_model_files",
                "OrpheusTTS",
            }

            ref_audio_path = (
                st.session_state.get("fine_tune_sample_path", "")
                if is_finetuned_model
                else sample_audio_path
            )

            metrics = run.evaluate_audio_metrics(
                audio_path,
                text_input,
                sample_audio_path=ref_audio_path,    # <<< passa o caminho correto
                lang="pt",
            )

            metrics_df = pd.DataFrame(
                {"Metrica": metrics.keys(), "Pontuação": metrics.values()}
            )

            metrics_df.index = metrics_df.index + 1 

            st.subheader("Métricas de qualidade de áudio")
            st.table(metrics_df)
    else:
        st.warning("Por favor, insira um texto para gerar o áudio.")