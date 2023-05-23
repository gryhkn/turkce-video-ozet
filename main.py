import os
import streamlit as st
from llama_index import download_loader, GPTVectorStoreIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter
import json
import datetime
from html2image import Html2Image


def set_up_st():
    st.set_page_config(page_title="Türkçe Video Özet")
    st.title("Özet Geç")
    st.caption("Bu uygulama ile Türkçe youtube videolarının özetlerini oluşturabilirsiniz. "
               "Eğer video 5 dakikadan uzunsa, her 5 dakikanın özetini çıkarır. Kullanmak için OpenAI API keyinizi ve özetlemek istediğiniz video linkini girmeniz gerekiyor... ")
    st.caption(
        "Daha fazla AI uygulaması için beni [Giray](https://twitter.com/gryhkn) takip etmeyi unutmayın.")

def check_api_key(api_key):
    if not api_key.startswith('sk-'):
        st.error('Yanlış anahtar girdiniz!')
        st.stop()
    else:
        st.success('Doğru anahtar!')


def send_click(youtube_link):
    st.session_state.video_id = youtube_link.split("v=")[1][:11]


def main():
    doc_path = './storage/'
    transcript_file = './storage/transkript.json'
    youtube_img = 'video_ss.png'
    youtube_link = ''
    if 'video_id' not in st.session_state:
        st.session_state.video_id = ''
    api_key = st.text_input('OpenAI API Key Girin:')
    check_api_key(api_key)
    youtube_link = st.text_input("Youtube linki:")
    st.button("Özetle!", on_click=lambda: send_click(youtube_link))
    if st.session_state.video_id != '':
        process_video(api_key, youtube_link, doc_path, transcript_file, youtube_img)


def process_video(api_key, youtube_link, doc_path, transcript_file, youtube_img):
    progress_bar = st.progress(5, text=f"Özetleniyor...")

    srt = YouTubeTranscriptApi.get_transcript(st.session_state.video_id, languages=['tr'])
    formatter = JSONFormatter()
    json_formatted = formatter.format_transcript(srt)
    with open(transcript_file, 'w') as f:
        f.write(json_formatted)

    hti = Html2Image()
    hti.screenshot(url=f"https://www.youtube.com/watch?v={st.session_state.video_id}", save_as=youtube_img)

    SimpleDirectoryReader = download_loader("SimpleDirectoryReader")

    loader = SimpleDirectoryReader(doc_path, recursive=True, exclude_hidden=True)
    documents = loader.load_data()

    # define LLM
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(openai_api_key=api_key, temperature=0, model_name="gpt-3.5-turbo", max_tokens=500))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    index = GPTVectorStoreIndex.from_documents(
        documents, service_context=service_context
    )

    section_texts = ''
    section_start_s = 0

    with open(transcript_file, 'r') as f:
        transcript = json.load(f)

    start_text = transcript[0]["text"]

    progress_steps = int(transcript[-1]["start"] / 300 + 2)
    progress_period = int(100 / progress_steps)
    progress_timeleft = str(datetime.timedelta(seconds=20 * progress_steps))
    percent_complete = 5
    progress_bar.progress(percent_complete, text=f"Özetleniyor...{progress_timeleft} kaldı")

    section_response = ''
    section_summary = ''

    # Assistant should continue the provided code
    for d in transcript:

        if d["start"] <= (section_start_s + 300) and transcript.index(d) != len(transcript) - 1:
            section_texts += ' ' + d["text"]

        else:
            end_text = d["text"]

            prompt = f"Bu metni \'{start_text}\' dan \'{end_text}\'ya kadar 500 kelime ile özetle, \'Videonun bu kısmı\' diye başla"
            query_engine = index.as_query_engine()
            response = query_engine.query(prompt)

            start_time = str(datetime.timedelta(seconds=section_start_s))
            end_time = str(datetime.timedelta(seconds=int(d['start'])))

            section_start_s += 300
            start_text = d["text"]
            section_texts = ''

            section_response += f"**{start_time} - {end_time}:**\n\r{response}\n\r"
            section_summary += f"**{start_time} - {end_time}:**\n\r{response}\n\r"

            percent_complete += progress_period
            progress_steps -= 1
            progress_timeleft = str(datetime.timedelta(seconds=20 * progress_steps))
            progress_bar.progress(percent_complete, text=f"Özetleniyor...{progress_timeleft} kaldı")

    progress_bar.progress(100, text="Tamamlandı!")
    st.subheader("Özet:")
    st.image(youtube_img)
    st.success(section_summary, icon="🤖")

    st.session_state.video_id = ''
    st.stop()


if __name__ == "__main__":
    set_up_st()
    main()
