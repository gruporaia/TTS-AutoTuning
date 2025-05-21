import os

# Caminhos
run_path = "/store/lucasouza/projetos/sintese_de_voz/data/Testando"  # ← ajuste para o seu modelo
output_dir = "/store/lucasouza/projetos/sintese_de_voz/scripts/metrics/finetune_output"
script_path = "/store/lucasouza/projetos/sintese_de_voz/scripts/xTTS-v2/synthesize.py"  # ← caminho até o generate_audio.py

# Garante que a pasta de saída existe
os.makedirs(output_dir, exist_ok=True)

# Frases em português com mais de 20 palavras
sentences = [
    "A tecnologia de síntese de voz tem evoluído muito nos últimos anos, permitindo criar áudios cada vez mais realistas e naturais.",
    "Mesmo com avanços impressionantes, ainda há desafios relacionados à expressividade, entonação e adaptação ao contexto específico do conteúdo.",
    "Avaliar a qualidade de modelos de voz requer métricas objetivas e subjetivas, envolvendo análise humana e automatizada do conteúdo gerado.",
    "Em experimentos com TTS, é fundamental garantir que as sentenças sejam diversificadas em vocabulário e estrutura para uma análise mais precisa.",
    "O português apresenta particularidades fonéticas e prosódicas que diferem de outras línguas, o que torna sua síntese mais complexa.",
    "Muitos modelos atuais são treinados principalmente em inglês, o que pode afetar negativamente a qualidade das versões em português.",
    "A aplicação de modelos de TTS em áreas como educação e acessibilidade demonstra o impacto social positivo da tecnologia.",
    "Ao analisar a naturalidade do áudio gerado, é importante considerar pausas, ritmo, ênfase nas palavras e clareza na articulação."
]

# Laço para gerar os áudios
for i, sentence in enumerate(sentences, 1):
    if(i < 2): continue


    output_path = os.path.join(output_dir, f"audio_{i}.wav")
    command = f'python "{script_path}" "{run_path}" "{output_path}" "{sentence}"'
    print(f"Executando: {command}")
    os.system(command)
    break