import streamlit as st
import yt_dlp
import os
import shutil
from tempfile import TemporaryDirectory
from moviepy.editor import AudioFileClip
from groq import Groq
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
# Importar o componente do Markmap
from streamlit_markmap import markmap

# Configuração Groq
groq_api_key = ""
whisper_model = 'whisper-large-v3-turbo'


def download_youtube_audio(url):
    with TemporaryDirectory() as temp_dir:
        audio_output_path = os.path.join(temp_dir, "downloaded_audio.mp3")
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(temp_dir, 'downloaded_audio.%(ext)s')
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        if os.path.exists(audio_output_path):
            # Move o arquivo para um local permanente
            permanent_path = os.path.join(os.getcwd(), "audio.mp3")
            shutil.move(audio_output_path, permanent_path)
            return permanent_path
        else:
            raise FileNotFoundError(f"O arquivo de áudio não foi criado em: {audio_output_path}")


def transcribe_audio_with_groq(filepath):
    client = Groq(api_key=groq_api_key)
    if os.path.exists(filepath):
        with open(filepath, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(filepath, file.read()),
                model=whisper_model,
            )
        return transcription.text
    else:
        raise FileNotFoundError(f"Arquivo de áudio não encontrado: {filepath}")


def summarize_text_with_llama(text, option):
    # Validar o tipo de texto
    if not isinstance(text, str):
        raise TypeError(f"O texto fornecido não é uma string válida: {type(text)}")
    # Validar as opções disponíveis
    if option not in ["Resumo", "Mapa Mental"]:
        raise ValueError(f"Opção inválida fornecida: {option}")

    # Configuração e prompt para o modelo
    if option == "Mapa Mental":
        system_prompt = """Você é um assistente especializado em criar mapas mentais.
        Crie um mapa mental em formato Markdown usando títulos e listas.
        Estruture o conteúdo de forma hierárquica usando # para títulos e - para listas.
        Use no máximo 3 níveis de profundidade para manter a clareza.

        Exemplo do formato esperado:
        # Tema Central
        ## Tópico Principal 1
        - Subtópico 1.1
            - Detalhe 1.1.1
        - Subtópico 1.2
        ## Tópico Principal 2
        - Subtópico 2.1
        - Subtópico 2.2
        """
    else:
        system_prompt = "Você é um assistente que cria resumos concisos e bem estruturados."

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"""
            Texto de entrada: {text}
            Tipo de saída: {option}

            Se for um resumo, forneça um texto conciso e bem estruturado.
            Se for um mapa mental, forneça em formato de Markdown hierárquico conforme especificado.
            """
        }
    ]

    chat = ChatGroq(
        temperature=0.4,
        model_name="llama-3.2-90b-text-preview",
        groq_api_key=groq_api_key
    )

    try:
        response = chat.invoke(messages)
        return response.content
    except Exception as e:
        raise RuntimeError(f"Erro ao processar o texto com o modelo Llama: {e}")


# Interface do Streamlit
st.title("Transcrição e Análise de Vídeos do YouTube")
youtube_url = st.text_input("Insira o link do vídeo do YouTube")

if youtube_url:
    st.write("Baixando e processando o áudio...")
    try:
        audio_file = download_youtube_audio(youtube_url)
        st.write("Transcrevendo o áudio com Whisper Large v3 Turbo...")
        transcribed_text = transcribe_audio_with_groq(audio_file)
        st.write("Texto transcrito:")
        st.text_area("Transcrição", transcribed_text, height=300)

        option = st.selectbox("Escolha o tipo de saída:", ["Selecione...", "Resumo", "Mapa Mental"])

        if option != "Selecione...":
            st.write("Processando com o modelo Llama...")
            output = summarize_text_with_llama(transcribed_text, option)

            if option == "Mapa Mental":
                # Renderiza o mapa mental usando Markmap
                st.write("Renderizando o mapa mental com Markmap...")
                st.markdown("### Mapa Mental")
                markmap(output)
            else:
                st.text_area("Resumo gerado:", output, height=200)
    except FileNotFoundError as e:
        st.error(f"Erro ao processar o áudio: {e}")