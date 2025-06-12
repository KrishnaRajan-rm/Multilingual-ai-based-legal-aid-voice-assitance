from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import speech_recognition as sr
from deep_translator import GoogleTranslator
from gtts import gTTS
import io
import os
import pygame
import requests
from neo4j import GraphDatabase

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (replace "*" with your frontend URL in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Initialize speech recognizer and pygame for audio playback
r = sr.Recognizer()
pygame.mixer.init()

# API and database credentials
gemini_api_key = "AIzaSyDKw6iYOdHWeZDonzfMIEnn7vykHUnWs4k"
neo4j_uri = "neo4j+s://691ef1aa.databases.neo4j.io"
neo4j_user = "neo4j"
neo4j_password = "ygDVA8lxN2Dn-ZuDeawDjZaivNOUO973-sS9vPD5_D0"

# Language code mappings
LANGUAGES = {
    "ta": "Tamil", "te": "Telugu", "hi": "Hindi", "ml": "Malayalam",
    "bn": "Bengali", "gu": "Gujarati", "kn": "Kannada", "mr": "Marathi",
    "pa": "Punjabi", "ur": "Urdu", "as": "Assamese", "or": "Odia",
    "kok": "Konkani", "mai": "Maithili", "sat": "Santali", "ne": "Nepali",
    "sd": "Sindhi", "ks": "Kashmiri", "doi": "Dogri", "mni": "Manipuri",
    "sa": "Sanskrit", "brx": "Bodo", "tcy": "Tulu", "en": "English"
}
LANGUAGE_CODES = {v: k for k, v in LANGUAGES.items()}

class GeminiAPI:
    """Handles requests to Gemini AI API."""
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

    def ask(self, prompt):
        headers = {"Content-Type": "application/json"}
        data = {"contents": [{"parts": [{"text": prompt}]}]}
        response = requests.post(f"{self.base_url}?key={self.api_key}", headers=headers, json=data)
        if response.status_code == 200:
            return response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No response received.")
        return f"Error: API request failed with status {response.status_code}: {response.text}"

class Neo4jConnector:
    """Manages Neo4j database connections."""
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def query(self, query):
        try:
            with self.driver.session() as session:
                result = session.run(query)
                return [record.data() for record in result] if result else None
        except Exception as e:
            return None  # Prevents breaking flow

class LegalAidAIAssistant:
    """AI-powered legal assistant using Gemini and Neo4j."""
    def __init__(self, gemini_api_key, neo4j_uri, neo4j_user, neo4j_password):
        self.gemini = GeminiAPI(gemini_api_key)
        self.neo4j = Neo4jConnector(neo4j_uri, neo4j_user, neo4j_password)

    def generate_answer(self, question):
        system_prompt = """
        You are a Legal AI Assistant specializing in Indian law.
        Start with a brief legal summary and ask if the user wants detailed steps.
        If no data is found in Neo4j, answer with pre-trained legal knowledge.
        give this in copy paste format no bold letter words.
        """
        cypher_prompt = f"Convert the legal question to a Neo4j Cypher query: {question}"
        cypher_query = self.gemini.ask(cypher_prompt)
        knowledge = self.neo4j.query(cypher_query)
        final_prompt = f"{system_prompt}\nUser: {question}\nNeo4j Data: {knowledge or 'None'}\nAssistant:"
        return self.gemini.ask(final_prompt)

    def close(self):
        self.neo4j.close()

# Define the request model for /ask_legal/
class QuestionRequest(BaseModel):
    question: str

@app.post("/ask_legal/")
def ask_legal(request: QuestionRequest):
    """Handles legal questions using Neo4j and Gemini AI."""
    try:
        legal_ai = LegalAidAIAssistant(gemini_api_key, neo4j_uri, neo4j_user, neo4j_password)
        response = legal_ai.generate_answer(request.question)
        legal_ai.close()
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribes speech to text."""
    try:
        with open("temp.wav", "wb") as buffer:
            buffer.write(await file.read())
        with sr.AudioFile("temp.wav") as source:
            audio = r.record(source)
        text = r.recognize_google(audio)
        os.remove("temp.wav")
        return {"transcription": text}
    except Exception as e:
        os.remove("temp.wav")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/translate/")
def translate_text(text: str = Query(...), target_lang: str = Query(...)):
    """Translates text into the specified language."""
    try:
        translated_text = GoogleTranslator(source="auto", target=target_lang).translate(text)
        return {"translated_text": translated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/generate_speech/")
def generate_speech(text: str = Query(...), lang: str = Query(...)):
    """Converts text to speech and plays audio."""
    try:
        play_tts_in_chunks(text, lang)
        return {"message": "Speech generated and played successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def play_tts_in_chunks(text, lang):
    """Splits long text into smaller chunks and plays audio."""
    sentences = text.split('. ')
    for chunk in sentences:
        tts = gTTS(text=chunk, lang=lang, slow=False)
        audio_data = io.BytesIO()
        tts.write_to_fp(audio_data)
        audio_data.seek(0)
        temp_file = "temp_audio.mp3"
        with open(temp_file, "wb") as f:
            f.write(audio_data.read())
        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        os.remove(temp_file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8080)
