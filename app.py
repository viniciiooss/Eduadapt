import streamlit as st
import yt_dlp
import re
import shutil
from tempfile import TemporaryDirectory
from groq import Groq
from langchain_groq import ChatGroq
from streamlit_markmap import markmap
import streamlit.components.v1 as components
from dotenv import load_dotenv
import os

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Obter a chave de API do ambiente
groq_api_key = os.getenv("GROQ_API_KEY")

# Validar se a chave de API foi carregada
if not groq_api_key:
    st.error("Erro: A chave de API do Groq não foi encontrada. Verifique as configurações do ambiente.")

whisper_model = 'whisper-large-v3'

def validate_youtube_url(url):
    """Valida se a URL é do YouTube"""
    youtube_regex = r'^(https?\:\/\/)?(www\.youtube\.com|youtu\.?be)\/.+$'
    return re.match(youtube_regex, url) is not None


def download_youtube_audio(url):
    """Baixa o áudio do vídeo do YouTube"""
    with TemporaryDirectory() as temp_dir:
        audio_output_path = os.path.join(temp_dir, "downloaded_audio.mp3")

        # Configurações com cookies do navegador
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(temp_dir, 'downloaded_audio.%(ext)s'),
            'cookiesfrombrowser': ('chrome',),  # Tenta usar cookies do Chrome
            'quiet': False,
            'no_warnings': False,
            'extract_flat': False,
            'youtube_include_dash_manifest': False,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept-Language': 'en-us,en;q=0.5',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            },
            'cookiefile': 'youtube.com_cookies.txt'  # Arquivo para salvar cookies
        }

        try:
            # Criar arquivo de cookies básico
            cookie_data = '''# Netscape HTTP Cookie File
.youtube.com	TRUE	/	FALSE	0	CONSENT	YES+cb
.youtube.com	TRUE	/	FALSE	0	VISITOR_INFO1_LIVE	your_visitor_info
.youtube.com	TRUE	/	FALSE	0	YSC	your_ysc_value'''

            with open('youtube.com_cookies.txt', 'w') as f:
                f.write(cookie_data)

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                try:
                    st.info("Extraindo informações do vídeo...")
                    video_info = ydl.extract_info(url, download=True)

                    if not video_info:
                        raise Exception("Não foi possível extrair informações do vídeo")

                    video_title = video_info.get('title', 'Título não disponível')
                    video_duration = video_info.get('duration', 0)

                except Exception as e:
                    st.warning(f"Método padrão falhou, tentando alternativa...")

                    # Tentar com configurações alternativas
                    alt_opts = ydl_opts.copy()
                    alt_opts['format'] = 'worstaudio[ext=m4a]/worstaudio/worst'
                    alt_opts['force_generic_extractor'] = True

                    with yt_dlp.YoutubeDL(alt_opts) as ydl_alt:
                        video_info = ydl_alt.extract_info(url, download=True)
                        video_title = video_info.get('title', 'Título não disponível')
                        video_duration = video_info.get('duration', 0)

            # Verificar e mover o arquivo
            if os.path.exists(audio_output_path):
                permanent_path = os.path.join(os.getcwd(), "audio.mp3")
                shutil.move(audio_output_path, permanent_path)
                return permanent_path, video_title, video_duration
            else:
                # Procurar por qualquer arquivo de áudio
                audio_files = [f for f in os.listdir(temp_dir) if f.endswith(('.mp3', '.m4a', '.wav'))]
                if audio_files:
                    audio_path = os.path.join(temp_dir, audio_files[0])
                    permanent_path = os.path.join(os.getcwd(), "audio.mp3")
                    shutil.move(audio_path, permanent_path)
                    return permanent_path, video_title, video_duration

                raise Exception("Nenhum arquivo de áudio foi gerado")

        except Exception as e:
            st.error(f"Erro no download: {str(e)}")
            st.warning("O YouTube está solicitando verificação. Você pode tentar:")
            st.markdown("""
            1. Usar outro link do YouTube
            2. Tentar novamente mais tarde
            3. Verificar se o vídeo não tem restrições de idade ou região
            """)
            return None, None, None

        finally:
            # Limpar arquivo de cookies
            if os.path.exists('youtube.com_cookies.txt'):
                os.remove('youtube.com_cookies.txt')


def transcribe_audio_with_groq(filepath):
    """Transcreve o áudio usando Groq Whisper"""
    if not filepath or not os.path.exists(filepath):
        st.error(f"Arquivo de áudio não encontrado: {filepath}")
        return None

    try:
        client = Groq(api_key=groq_api_key)
        with open(filepath, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(filepath, file.read()),
                model=whisper_model,
            )
        return transcription.text
    except Exception as e:
        st.error(f"Erro na transcrição: {e}")
        return None


def summarize_text_with_llama(text, option, temperature=0.4):
    """Resumir ou criar mapa mental usando Llama"""
    if not text:
        st.error("Nenhum texto fornecido para processamento")
        return None

    # Configuração de prompt baseada na opção
    system_prompts = {
        "Mapa Mental": """Você é um assistente especializado em criar mapas mentais.
        Crie um mapa mental em formato Markdown usando títulos e listas.
        Estruture o conteúdo de forma hierárquica usando # para títulos e - para listas.
        Use no máximo 3 níveis de profundidade para manter a clareza.""",
        "Resumo": """Você é um assistente que cria resumos concisos e bem estruturados.
        Evite usar asteriscos ou marcadores especiais.
        Mantenha o texto limpo e direto.
        Use parágrafos para separar ideias diferentes."""
    }

    if option not in system_prompts:
        st.error(f"Opção inválida: {option}")
        return None

    messages = [
        {"role": "system", "content": system_prompts[option]},
        {"role": "user", "content": f"Texto de entrada: {text}\nTipo de saída: {option}"}
    ]

    try:
        chat = ChatGroq(
            temperature=temperatura,
            model_name=modelo_llm,
            groq_api_key=groq_api_key
        )
        response = chat.invoke(messages)
        return response.content
    except Exception as e:
        st.error(f"Erro ao processar texto: {e}")
        return None


# Configuração da página Streamlit
st.set_page_config(page_title="EduAdapt", page_icon="🎬", layout="wide")
header_col1, header_col2 = st.columns([0.1, 0.9])

with header_col1:
    # Substitua "nova_imagem.png" pelo caminho da sua imagem menor
    st.image("img.png", width=70)

with header_col2:
    st.markdown(
        """
        <div style="display: flex; align-items: center; height: 100%;">
            <h1 style="font-size: 38px; margin-top:-10px; margin-left: -30px;">EduAdapt</h1>
        </div>
        """,
        unsafe_allow_html=True
    )


# Textos abaixo
st.title("🎥 Transcrição e Análise de Vídeos do YouTube")
st.markdown("Transforme vídeos do YouTube em resumos e mapas mentais!")

# Sidebar de configurações
st.sidebar.header("⚙️ Configurações")
temperatura = st.sidebar.slider("Temperatura do Modelo", 0.0, 1.0, 0.4,
                                help="Controla a criatividade do modelo. Valores mais altos geram respostas mais variadas.")
modelo_llm = st.sidebar.selectbox("Modelo LLM",
                                  ["llama-3.2-90b-vision-preview",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "llama3-groq-70b-8192-tool-use-preview",
    "llama3-70b-8192"])

# Campo de entrada de URL
youtube_url = st.text_input("📋 Insira o link do vídeo do YouTube")

# Botão de limpeza
if st.sidebar.button("🔄 Limpar Tudo"):
    youtube_url = ""
    st.experimental_rerun()


# Processamento principal
if youtube_url:
    # Validação da URL
    if not validate_youtube_url(youtube_url):
        st.warning("⚠️ Por favor, insira uma URL válida do YouTube")
    else:
        # Progress bar para feedback
        progress_bar = st.progress(0)

        try:
            # Download do áudio
            progress_bar.progress(25)
            st.write("🔊 Baixando áudio...")
            audio_file, video_title, video_duration = download_youtube_audio(youtube_url)

            if audio_file:
                # Exibir informações do vídeo
                st.write(f"📹 **Título:** {video_title}")
                st.write(f"⏱️ **Duração:** {video_duration} segundos")

                # Transcrição
                progress_bar.progress(50)
                st.write("📝 Transcrevendo áudio...")
                transcribed_text = transcribe_audio_with_groq(audio_file)

                if transcribed_text:
                    col_trans_title, col_trans_button = st.columns([0.9, 0.1])
                    with col_trans_title:
                        st.write("📄 Transcrição Completa")
                    with col_trans_button:
                        show_transcription = st.button("👁️")

                    if show_transcription:
                        st.text_area("", transcribed_text, height=250)

                    # Opções de processamento
                    progress_bar.progress(75)
                    option = st.selectbox("Escolha o tipo de saída:",
                                          ["Selecione...", "Resumo", "Mapa Mental"])

                    if option != "Selecione...":
                        st.write(f"🧠 Processando {option}...")
                        output = summarize_text_with_llama(transcribed_text, option, temperatura)

                        if output:
                            progress_bar.progress(100)
                            if option == "Mapa Mental":
                                st.markdown("### 🗺️ Mapa Mental")
                                # Dividir a área em duas colunas para colocar o botão ao lado
                                col_output, col_button = st.columns([0.85, 0.15])

                                with col_output:
                                    markmap(output)

                                with col_button:
                                    # Botão para baixar o mapa mental em formato Markdown
                                    mapa_filename = "mapa_mental.md"
                                    st.download_button(
                                        label="⬇️",
                                        data=output,
                                        file_name=mapa_filename,
                                        mime="text/markdown"
                                    )

                            else:
                                st.markdown(f"### 📋 {option} Gerado:")

                                # Dividir a área em duas colunas para colocar o botão ao lado
                                col_output, col_button = st.columns([0.85, 0.15])

                                with col_output:
                                 paragraphs = output.split('\n')

                                # Definir o HTML completo com estilo e parágrafos
                                 html_output = """
                                      <!DOCTYPE html>
                                      <html>
                                      <head>
                                          <style>
                                              .colored-paragraph {
                                                  padding: 15px;
                                                  margin: 10px 0;
                                                  border-radius: 10px;
                                                  font-family: Arial, sans-serif;
                                              }
                                          </style>
                                      </head>
                                      <body>
                                      """

                                # Lista de cores de fundo e cor do texto para os parágrafos
                                 colors = [
                                    {'bg_color': '#D32F2F', 'text_color': '#FFFFFF'},  # Vermelho
                                    {'bg_color': '#C2185B', 'text_color': '#FFFFFF'},  # Rosa
                                    {'bg_color': '#7B1FA2', 'text_color': '#FFFFFF'},  # Roxo
                                    {'bg_color': '#512DA8', 'text_color': '#FFFFFF'},  # Roxo escuro
                                    {'bg_color': '#303F9F', 'text_color': '#FFFFFF'},  # Índigo
                                    {'bg_color': '#1976D2', 'text_color': '#FFFFFF'},  # Azul
                                    {'bg_color': '#0288D1', 'text_color': '#FFFFFF'},  # Azul claro
                                    {'bg_color': '#0097A7', 'text_color': '#FFFFFF'},  # Ciano
                                    {'bg_color': '#00796B', 'text_color': '#FFFFFF'},  # Verde azulado
                                    {'bg_color': '#388E3C', 'text_color': '#FFFFFF'},  # Verde
                                    {'bg_color': '#689F38', 'text_color': '#FFFFFF'},  # Verde oliva
                                    {'bg_color': '#AFB42B', 'text_color': '#FFFFFF'},  # Lima
                                    {'bg_color': '#FBC02D', 'text_color': '#FFFFFF'},  # Amarelo
                                    {'bg_color': '#FFA000', 'text_color': '#FFFFFF'},  # Âmbar
                                    {'bg_color': '#F57C00', 'text_color': '#FFFFFF'},  # Laranja
                                    {'bg_color': '#E64A19', 'text_color': '#FFFFFF'},  # Laranja escuro
                                    {'bg_color': '#5D4037', 'text_color': '#FFFFFF'},  # Marrom
                                    {'bg_color': '#616161', 'text_color': '#FFFFFF'},  # Cinza
                                    {'bg_color': '#455A64', 'text_color': '#FFFFFF'},  # Azul acinzentado

                                ]

                                # Gerar o HTML com os parágrafos estilizados
                                 for i, para in enumerate(paragraphs):
                                    if para.strip():
                                        color = colors[i % len(colors)]
                                        bg_color = color['bg_color']
                                        text_color = color['text_color']
                                        html_output += f"""
                                              <div class="colored-paragraph" style="background-color: {bg_color}; color: {text_color};">
                                                  <p>{para}</p>
                                              </div>
                                              """

                                # Fechar o body e o html
                                 html_output += """
                                      </body>
                                      </html>
                                      """

                                # Renderizar o HTML no Streamlit usando components.html
                                components.html(html_output, height=2000, scrolling=False)
                                with col_button:
                                    # Botão para baixar o resumo em formato de texto
                                    resumo_filename = "resumo.txt"
                                    st.download_button(
                                        label="⬇️",
                                        data=output,
                                        file_name=resumo_filename,
                                        mime="text/plain"
                                    )


        except Exception as e:
            st.error(f"Erro no processamento: {e}")
            st.info("Verifique a URL ou tente novamente")

        finally:
            # Limpar barra de progresso
            progress_bar.empty()

# Rodapé informativo
st.sidebar.markdown("---")
st.sidebar.info("Desenvolvido com Streamlit, Whisper, Groq e Llama")