import streamlit as st
import time
import run_scripts as run
import os
import csv
import zipfile
import pandas as pd
from TTS_Utils import UTMOS, SECS, CER, build_dataset, normalize_text

sys.path.append(os.path.abspath('../scripts/orpheusTTS'))
from pre_trained import synthesize as orpheus_pre_trained

# ─────────────── page config ───────────────
st.set_page_config(
    page_title="Fine-tune & Generate Audio",
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


# criar csv com dados do modelos
# with open("models.csv", "w") as f:
#     csv_writer = csv.writer(f, delimiter=",")
#     csv_writer.writerow(["nome", "path", "model_type", "score"])
#     pasta_modelos = "../data/modelos"
#     os.makedirs(pasta_modelos, exist_ok=True)
#     models = [
#         (nome, 
#         os.path.join(pasta_modelos, nome), 
#         "xTTs-v2" if any(f.endswith(".pth") for f in os.listdir(os.path.join(pasta_modelos, nome))) else "orpheusTTS", # confere se é um modelo xtts ou orpheus dependendo da terminação do arquivo do modelo
#         0.0)
#         for nome in os.listdir(pasta_modelos)
#     ]
#     for modelo in models:
#         csv_writer.writerow(modelo)
    

# ─────────────── UI ───────────────
###### Fine-tuning
st.title("Fine-tune & Generate Audio Demo")
st.divider()

st.subheader("Fine-tune Model")
uploaded_files = st.file_uploader(
    "Envie arquivos de áudio para ajuste fino:",
    type=["wav", "mp3", ".zip"],
    accept_multiple_files=True,
    key="fine_tune_files",
)

avaliables_models = {
    "Modelo xTTS-v2 (PT-BR)": "xTTS-v2",
    "Modelo OrpheusTTS (EN)": "orpheusTTS"
}

avaliables_durations = {
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
    "Digite o nome do falante:",
    placeholder="Ex.: Clarice Lispector",
)

model_to_tuning = st.radio("Selecione o modelo para fine-tuning", avaliables_models.keys())
duration_to_tuning = st.radio("Selecione o número de épocas/steps para fine-tuning", avaliables_durations.keys())
learning_to_tuning = st.radio("Selecione a taxa de aprendizado", avaliables_learning.keys())

if st.button("Iniciar Ajuste Fino", key="fine_tune"):
    print("-------------------------------------------")
    print("-------------------------------------------")
    print("-------------------------------------------")
    print("-------------------------------------------")
    print("-------------------------------------------")
    print("-------------------------------------------")
    print("-------------------------------------------")
    print("-------------------------------------------")
    print("-------------------------------------------")
    print("-------------------------------------------")
    print("-------------------------------------------")
    print("-------------------------------------------")
    if uploaded_files:
        input_audio_path = '../data/raw_audio'
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

        with st.spinner("Processando áudio…"):
            #build_dataset(input_dir='../data/raw_audio', output_dir='../data/audio_transcription')
            st.success("Áudio processado")

        with st.spinner("Fazendo ajuste fino no modelo"):
            print(run.finetune(speaker_name, avaliables_models[model_to_tuning], avaliables_durations[duration_to_tuning], avaliables_learning[learning_to_tuning], inputType))
            st.success("Ajuste fino concluído!")
    else:
        st.warning("Por favor, envie arquivos de áudio antes de iniciar o ajuste fino.")

st.divider()

###### Inferência
st.subheader("Gerar Áudio")
text_input = st.text_area(
    "Digite o texto para gerar o áudio:",
    placeholder="Ex.: Olá, seja bem-vindo…",
    key="text_to_generate",
)

models = {}
models_type = {} # Se o modelo é o xtts ou orpheus
#model_evals = {}

pasta_modelos = "../data/modelos"

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

#carrega opções de modelo do csv
# with open("models.csv") as csv_modelos:
#     csv_reader = csv.reader(csv_modelos, delimiter=',')
#     next(csv_reader) #cabecalho
#     for row in csv_reader:
#         models[row[0].strip()] = row[1].strip()
#         models_type[row[0].strip()] = row[2].strip()
#         #model_evals[row[0].strip()] = row[3].strip()


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
        input_audio_path = '../data/sample_audio'

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
        sample_audio_path = '../data/XTTS_v2.0_original_model_files'
        sample_audio_path = os.path.join(sample_audio_path, xtts_audio_sample.name)
        sample_byte = xtts_audio_sample.getvalue()
        with open(sample_audio_path, "wb") as f:
            f.write(sample_byte)

if st.button("Gerar Áudio", key="generate_audio"):
    audio_path = ""
    if text_input.strip():
        with st.spinner("Gerando áudio…"):
            fake_progress()
            normalized_text = normalize_text(text_input)

            print(model_select)

            # Orpheus Pré-treinado
            if model_select == "OrpheusTTS" and sample_audio_path:
                output_path = '../data/gen'
                audio_path = orpheus_pre_trained(sample_audio_path, output_path, normalized_text, normalized_transcript)
            # xTTS pré-treinado
            elif model_select == "XTTS_v2.0_original_model_files" and sample_audio_path: # garante que a amostra foi enviada
                audio_path = run.synthesize(normalized_text, models[model_select], models_type[model_select])
            # Demais modelos fine-tunados
            elif model_select != "XTTS_v2.0_original_model_files":
                audio_path = run.synthesize(normalized_text, models[model_select], models_type[model_select])

        if audio_path:
            with open(audio_path, "rb") as audio_arquivo:
                audio_bytes = audio_arquivo.read()

            metrics = run.evaluate_audio_metrics(
                audio_path,
                text_input,
                sample_audio_path=sample_audio_path,
                lang="pt",
            )

            metrics_df = pd.DataFrame(
                {"Metric": metrics.keys(), "Score": metrics.values()}
            )

            st.subheader("Audio-quality metrics")
            st.table(metrics_df)
    else:
        st.warning("Por favor, insira um texto para gerar o áudio.")