import streamlit as st
import asyncio
from ollama import AsyncClient
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import os
import speech_recognition as sr
import pyttsx3
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization, DepthwiseConv2D
import time

class CustomDepthwiseConv2D(DepthwiseConv2D):
    def _init_(self, *args, **kwargs):
        kwargs.pop('groups', None) 
        super()._init_(*args, **kwargs)

os.environ["PINECONE_API_KEY"] = "<PINECONE API>"

try:
    model = load_model(
        "keras2222/keras_Model.h5",
        compile=False,
        custom_objects={
            'BatchNormalization': BatchNormalization,
            'DepthwiseConv2D': CustomDepthwiseConv2D
        }
    )
    st.write("Model loaded successfully.")
except Exception as e:
    st.error("Error loading model: {}".format(e))
    exit()

try:
    with open("keras2222/labels.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    st.write("Labels loaded successfully.")
except Exception as e:
    st.error("Error loading labels: {}".format(e))
    exit()

loader = PyPDFLoader("ncert_test_file.pdf")
st.html(r"hackhome.html")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=0)
docs = text_splitter.split_documents(data)

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
index_name = "orkho"
docsearch = Pinecone.from_texts([t.page_content for t in docs], embeddings, index_name=index_name)

recognizer = sr.Recognizer()

def speech_to_text():
    with sr.Microphone() as source:
        st.write("Listening... Please speak clearly.")
        recognizer.adjust_for_ambient_noise(source)  
        audio = recognizer.listen(source)
        
        try:
            text = recognizer.recognize_google(audio)
            st.success(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Could not understand audio. Please try again.")
            return ""
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service; {e}")
            return ""

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def webcam_hand_sign_prediction():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        st.error("Error: Could not open camera.")
        return None

    ret, image = camera.read()
    camera.release()

    if not ret:
        st.error("Failed to grab frame")
        return None

    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
    image_array = (image_array / 127.5) - 1 

    prediction = model.predict(image_array)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score

def mm():
    t=""
    while True:
        result = webcam_hand_sign_prediction()
        if result:
            class_name, confidence_score = result
            # t+=class_name[2:]+" "
            st.write("Sign input: {}".format(class_name[2:]))
            # st.write("Confidence Score: {:.2f}%".format(confidence_score * 100))
            time.sleep(2)
            if class_name.strip() == "2 focus": t+="what is focus of lens"
            elif class_name.strip()=="1 reflection": t+="what is reflection"
            elif class_name.strip() == "6 none":break
            
    st.write(f"Input is {t}")

    d = docsearch.similarity_search(t)

    async def chat():
        message = {
            "role": "user",
            "content": f"read {d} and just tell {t}"
        }

        message_parts = []
        
        async for part in await AsyncClient().chat(
            model="llama3.1", messages=[message], stream=True,
        ):
            message_parts.append(part["message"]["content"])

        full_message = " ".join(message_parts)
        st.write(full_message)

        text_to_speech(full_message)

    asyncio.run(chat())

    # return t


# Speech input
if st.button("Start Speaking"):
    user_inp = speech_to_text()
elif st.button("Capture Hand Sign"):
    mm()
# elif st.button("load Hand Sign"):
    # user_inp=t
else:
    user_inp = st.text_input("Enter Question Here:")

if st.button("Search") or user_inp :
    query = user_inp

    if query:  
        d = docsearch.similarity_search(query)

        async def chat():
            message = {
                "role": "user",
                "content": f"read {d} and just tell {query}"
            }

            message_parts = []
            
            async for part in await AsyncClient().chat(
                model="llama3.1", messages=[message], stream=True,
            ):
                message_parts.append(part["message"]["content"])

            full_message = " ".join(message_parts)
            st.write(full_message)

            text_to_speech(full_message)

        asyncio.run(chat())
