# Automatizador de fine-tuning de modelos de síntese de voz para vozes únicas+
Modelos atuais permitem clonagem de voz de qualquer pessoa utilizando um áudio curto de referência. Entretanto, realizar finetuning do modelo para aquela voz, com mais exemplos, leva a uma clonagem mais realista. Pensando nisso, este projeto propõe-se a facilitar esse processo ao desenvolver uma automação de fine-tuning para modelos de TTS. O projeto conta com:
- Gerador de datasets a partir de áudio brutos em língua portuguesa.
- Normalizador de textos em língua portuguesa. (Aplicação que expande abreviações e reescreve símbolos de maneira extensa para que o modelo de voz seja capaz de transformar em áudio de maneira correta).
- Avaliação dos resultados dos modelos através de métricas 
- Exemplo de fine-tuning com o modelo XTTS-v2 em língua portuguesa.
- Exemplo de fine-tuning com o modelo Orpheus-TTS em língua inglesa.
- Interface interativa desenvolvida em Streamlit, que pode ser usada para manipular, em alto nível, os componentes citados acima.

**Os três primeiros itens dessa lista estão presentes para uso direto da comunidade através do nosso repositório [TTS_Utils](https://github.com/gruporaia/TTS-Utils).**

## ⚙️ Funcionamento
A partir do nosso [TTS_Utils](https://github.com/gruporaia/TTS-Utils), conseguimos transformar qualquer conjunto de áudios não processados em um dataset pronto para ser utilizado para treinar um modelo de TTS. Utilizando-se disso, elaboramos um pipeline de finetuning automático, através de uma interface em Streamlit, de dois modelos de TTS: o [xTTS-v2](https://huggingface.co/coqui/XTTS-v2) e o [Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS).

Com os modelos treinados, permitimos, através da mesma interface, o uso deles para inferência.

![Image](https://github.com/user-attachments/assets/57b72fc1-cf98-4f85-8c1b-28526fb3dffe)

## 🛠️ Como rodar

### Clonando repositório
```bash
git clone https://github.com/gruporaia/TTS-AutoTuning.git
cd TTS-AutoTuning
```

### Instalando dependências

O repositório foi construído usando Python 3.11.11.

Usando pip:
```bash
pip install -r requirements.txt
```

ou usando miniconda:
```bash
conda env create -f environment.yml
```

_OBS: talvez seja necessário instalar a dependência 'FFmpeg'. Você pode fazer isso com conda ou apt install._

### Executando o projeto
```bash
cd app/
streamlit run app.py   #Abre a interface web
```

## 📊 Interface
![Image](https://github.com/user-attachments/assets/7c1f1769-9dc1-4298-ae93-3ef7fb271fbb)
![Image](https://github.com/user-attachments/assets/4d18c557-428e-4d61-9318-ba346733dbad)

### Próximos passos 
- Utilizar melhores modelos para clonagem de voz.
- Permitir maior autonomia para treinar os modelos através da aplicação, possibilitando variar mais parâmetros de maneira livre e retornando logs de treinamento na interface.


## 💻 Quem somos
| ![LogoRAIA](https://github.com/user-attachments/assets/ce3f8386-a900-43ff-af84-adce9c17abd2) |  Este projeto foi desenvolvido pelos membros do **RAIA (Rede de Avanço de Inteligência Artificial)**, uma iniciativa estudantil do Instituto de Ciências Matemáticas e de Computação (ICMC) da USP - São Carlos. Somos estudantes que compartilham o objetivo de criar soluções inovadoras utilizando inteligência artificial para impactar positivamente a sociedade. Para saber mais, acesse [nosso site](https://gruporaia.vercel.app/) ou [nosso Instagram](instagram.com/grupo.raia)! |
|------------------|-------------------------------------------|

Projeto feito em supervisão do Professor **Moacir Ponti** - [Site](https://sites.google.com/site/moacirponti/)

### Desenvolvedores
- **Antonio Carlos** - [LinkedIn](https://www.linkedin.com/in/ant%C3%B4nio-carlos-micheli-b10bb4289/) | [GitHub](https://github.com/Antonioonet)
- **Arthur Trottmann** - [LinkedIn](https://www.linkedin.com/in/arthur-ramos-9b81b9201/) | [GitHub](https://github.com/ArthurTRamos)
- **Caio Petroncini** - [LinkedIn](https://www.linkedin.com/in/caio-petroncini-7105941aa/) | [GitHub](https://github.com/Petroncini)
- **Lucas Brandão** - [LinkedIn](https://www.linkedin.com/in/lucas-de-souza-brandão-590b1228b/) | [GitHub](https://github.com/sb-lucas)
- **Pedro Soares** - [LinkedIn](https://www.linkedin.com/in/pedro-soares-b3625b238/) | [GitHub](https://github.com/pedrsrs)
