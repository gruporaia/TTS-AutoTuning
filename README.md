# Automatizador de fine-tuning de modelos de síntese de voz para vozes únicas

## 📌 Sobre o Projeto
Contratar famosos para usarem suas vozes em um projeto pode ser muito caro. No entanto, com as tecnologias de Machine Learning, é possível recriar vozes artificialmente, reduzindo custos, já que a empresa precisaria pagar apenas pelos direitos de uso, sem a necessidade de gravações diretas.

Somado a isso, estudos na área de Text-to-Speech (TTS) apontam que performar fine-tuning com dados da voz de uma única pessoa em um modelo de TTS resulta em melhores performances na recriação daquela voz em comparação a modelos generalistas.

Pensando nisso, este projeto propõe-se a facilitar esse modelo de negócio ao desenvolver uma automação de fine-tuning para modelos de TTS. O projeto conta com:
- Gerador de datasets a partir de áudio brutos em língua portuguesa.
- Exemplo de fine-tuning com o modelo XTTS-v2 em língua portuguesa.
- Exemplo de fine-tuning com o modelo Orpheus-TTS em língua inglesa.
- Normalizador de textos em língua portuguesa. (Aplicação que expande abreviações e reescreve símbolos de maneira extensa para que o modelo de voz seja capaz de transformar em áudio de maneira correta).
- Interface interativa desenvolvida em Streamlit, que pode ser usada para manipular, em alto nível, os componentes citados acima.

## 🚀 Como Rodar o Projeto

### 1️⃣ Clone o Repositório
```bash
git clone https://github.com/gruporaia/sintese-voz.git
cd sintese-voz
```

### 2️⃣ Instalar Dependências
🚨ATENÇÃO\
 Cada um dos componentes do projeto utiliza de seu próprio ambiente virtual com Miniconda a fim de gerenciar diferentes dependências.

#### Instalando Miniconda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

#### Environments

| Tarefa         | Arquivo de Ambiente               |
|----------------|-----------------------------------|
| Dataset        | `envs/dataset.yml`                |
| XTTS-v2        | `envs/xtts.yml`                   |
| OrpheusTTS     | `envs/orpheus.yml`                |
| Normalização   | `envs/norm.yml`                   |
| Aplicação      | `envs/app.yml`                    |

#### Criar Ambientes
Crie todos os ambientes listados acima.

```bash
conda env create -f envs/dataset.yml
conda env create -f envs/xtts.yml
conda env create -f envs/orpheus.yml
conda env create -f envs/norm.yml
conda env create -f envs/app.yml
```

### 3️⃣ Execute a Interface
```bash
cd app
conda activate app
streamlit run app.py
```

## 📷 Exemplo de Detecção
 ![Image](https://www.linkedin.com/in/lucas-de-souza-brandão-590b1228b/)

## Próximos Passos
- Adicionar pré-processador de datasets em outras línguas.
- Adicionar normalizador de textos em outras línguas.
- Desenvolver aplicação mais robusta usando os componentes apresentados.
 
---
 ## Organização responsável 
 ### RAIA - Rede de Avanço em Inteligência Artificial 
- [Antonio Carlos](https://www.linkedin.com/in/lucas-de-souza-brandão-590b1228b/) 
- [Arthur Trottmann](https://www.linkedin.com/in/lucas-de-souza-brandão-590b1228b/) 
- [Caio Petroncini](https://www.linkedin.com/in/lucas-de-souza-brandão-590b1228b/) 
- [Lucas Brandão](https://www.linkedin.com/in/lucas-de-souza-brandão-590b1228b/) 
- [Pedro Soares](https://www.linkedin.com/in/lucas-de-souza-brandão-590b1228b/) 
- 💡 Projeto em parceria com o [Professor Moacir Antonelli Ponti](https://www.linkedin.com/in/lucas-de-souza-brandão-590b1228b/)

