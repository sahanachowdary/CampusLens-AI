# 🎓 CampusLens AI

CampusLens AI is a multimodal Retrieval-Augmented Generation (RAG) application designed to help students access important college information from multiple sources such as PDF circulars, text notices, audio announcements, and image-based references.

Instead of manually searching through different notices or announcements, students can upload the available information and ask questions through a simple chat interface. CampusLens retrieves the most relevant information from the uploaded sources and generates a clear answer using a local LLM through Ollama.

The application also supports conversation memory and persistent chat history, allowing users to continue previous conversations even after restarting the application.

## 🚀 Features

- Upload and process PDF circulars and notices
- Add college announcements directly as text
- Upload audio announcements and convert speech into text using Whisper
- Upload and display images with manually added supporting information
- Convert text into vector embeddings using Sentence Transformers
- Store and retrieve information using ChromaDB
- Perform semantic similarity search
- Generate answers using Ollama and Llama 3.2
- Retrieval-Augmented Generation (RAG)
- Conversation memory for follow-up questions
- Persistent chat history using SQLite
- Create multiple chat sessions
- Reopen previous chats
- Delete individual chats
- Display the source used for generating answers
- View the retrieved context used by the RAG system
- Works locally without requiring a paid LLM API

## 🛠️ Tech Stack

- Python
- Streamlit
- PyPDF
- OpenAI Whisper
- FFmpeg
- Pillow
- Sentence Transformers
- ChromaDB
- Ollama
- Llama 3.2
- SQLite
- Retrieval-Augmented Generation (RAG)

## 🧠 How It Works

Different input formats are first converted into text.

PDF → PyPDF → Extracted Text

Audio → FFmpeg → Whisper → Transcript

Text Notice → Direct Text Input

Image → Display + Manual Description

The extracted text is then processed using the RAG pipeline:

Text
↓
Chunking
↓
Sentence Transformer Embeddings
↓
ChromaDB
↓
User Question
↓
Question Embedding
↓
Similarity Search
↓
Relevant Chunks
↓
Ollama + Llama 3.2
↓
Final Answer

Streamlit Session State is used for current conversation memory, while SQLite is used to store chat history permanently.

## 📁 Project Structure

CampusLens-AI/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── chroma_db/
└── chat_history.db

The `chroma_db` folder and `chat_history.db` file are generated automatically while running the project and are normally ignored in GitHub.

## ⚙️ Requirements

Recommended Python version:

Python 3.11

Recommended version used for development:

Python 3.11.9

Python packages required:

streamlit
pypdf
sentence-transformers
chromadb
ollama
openai-whisper
pillow

SQLite does not require a separate installation because the `sqlite3` module is already included with Python.

FFmpeg and Ollama must be installed separately on the system.

## 📦 Installation and Setup

Clone the repository:

git clone https://github.com/sahanachowdary/CampusLens-AI.git

Move into the project folder:

cd CampusLens-AI

Create a virtual environment:

py -3.11 -m venv .venv

Activate the virtual environment in PowerShell:

.venv\Scripts\Activate.ps1

Install the required Python packages:

python -m pip install -r requirements.txt

Check Python version:

python --version

Verify Ollama installation:

ollama --version

Download the Llama 3.2 1B model:

ollama pull llama3.2:1b

Check installed Ollama models:

ollama list

Verify FFmpeg installation:

ffmpeg -version

Run the application:

python -m streamlit run app.py

After running the command, Streamlit will open CampusLens AI in the browser.

## 💬 Example Usage

A user can add a college notice such as:

The Campus Innovation Hackathon will be conducted on October 15.
Registration closes on October 10.
Each team must contain 3 to 4 students.
Participants must carry their college ID cards.
The event will be held in the Main Auditorium.

The user can then ask:

When is the hackathon?

CampusLens retrieves the relevant information and generates the answer.

A follow-up question can be:

What is the team size?

Because conversation memory is enabled, CampusLens understands that the user is still referring to the hackathon.

## 🎧 Audio Support

CampusLens supports audio files such as MP3, WAV, M4A, and OPUS.

The uploaded audio is processed using FFmpeg and transcribed using Whisper.

For example, an audio announcement may contain:

The coding workshop will be conducted on October 20 in Lab 3.
Students should report at 10 AM.
Only second-year students are eligible.
Participants must bring their laptops and college ID cards.

After transcription, the information is stored in ChromaDB and can be queried using the chat interface.

Example question:

What should students bring for the coding workshop?

CampusLens retrieves the information from the audio transcript and provides the answer.

## 📄 PDF Support

CampusLens processes text-based PDF circulars using PyPDF.

The extracted text is divided into smaller chunks, converted into embeddings, and stored in ChromaDB.

Example questions include:

When is the internship orientation?

Where will it be conducted?

What documents should students bring?

The application also displays the PDF filename as the source of the answer.

## 🖼️ Image Support

CampusLens allows users to upload images such as posters or campus notices.

The uploaded image is displayed using Pillow.

Since automatic OCR is currently not enabled, users can manually add a short description of the image.

The description is stored in ChromaDB and becomes searchable through the RAG system.

Example:

Tech Fest will be conducted on November 2.
Registration closes on October 28.
The event will be held in Block A Auditorium.

The user can then ask:

When is the Tech Fest?

CampusLens retrieves the answer from the image description.

## 🧠 Conversation Memory

CampusLens maintains recent conversation context using Streamlit Session State.

This allows users to ask follow-up questions without repeating the topic.

Example:

User:
When is the hackathon?

CampusLens:
The hackathon will be conducted on October 15.

User:
What is the team size?

CampusLens:
Each team should contain 3 to 4 students.

User:
Where is it happening?

CampusLens:
The event will be held in the Main Auditorium.

## 💾 Persistent Chat History

CampusLens uses SQLite to store conversations permanently.

Users can:

- Start a new chat
- View previous chats
- Reopen old conversations
- Continue previous discussions
- Delete selected chats

The chat history is stored locally in:

chat_history.db

This makes the application behave more like a complete AI chat assistant instead of a temporary chatbot.

## 🔍 Source-Aware Responses

CampusLens displays the source from which the answer was retrieved.

For example:

Answer:
The internship orientation will be conducted on October 25.

Source:
Internship_Notice.pdf

Users can also open the retrieved context section to see the exact text chunks used by the RAG system.

## 🔐 Local AI and Privacy

CampusLens is designed to work mainly with local components.

- Ollama runs the LLM locally
- ChromaDB stores embeddings locally
- SQLite stores chat history locally
- Uploaded information does not require a paid cloud LLM API

This makes the project suitable for college information systems and local AI experimentation.

## ⚠️ Current Limitations

- Automatic OCR for images is not included
- Video input is not currently supported
- Voice-based questions are not yet implemented
- Performance depends on available RAM and CPU
- Whisper transcription time depends on the length of the audio file
- Streamlit Cloud deployment is limited because the project depends on local Ollama models and resource-intensive components such as Whisper and Torch.
These local services and high memory requirements are not reliably supported in the Streamlit Community Cloud environment.


## 🔮 Future Enhancements

- Voice-based question input
- Video upload and transcription
- Automatic image OCR
- Text-to-speech responses
- Automatic deadline extraction
- Deadline reminders
- Multilingual support
- User authentication
- Admin dashboard
- Improved mobile-friendly interface
- Deployment:- Replace the local Ollama model with a hosted LLM API such as Gemini or OpenAI for easier cloud deployment.
               Use a persistent cloud database/vector store such as Supabase or hosted ChromaDB for reliable storage.
               Deploy the optimized application on Streamlit Cloud or another cloud platform for public access.

## 🎯 Project Goal

The main goal of CampusLens AI is to make college information easier to access.

Students often receive important information through PDFs, audio announcements, text notices, and posters. CampusLens brings these different sources into a single AI-powered knowledge assistant and allows students to ask questions naturally.

Instead of manually searching through multiple sources, students can retrieve the required information through a simple conversational interface.

## 👩‍💻 Author

Parvthaneni Sahana Chowdari

B.Tech – Computer Science and Engineering

GitHub: @sahanachowdary

## 📄 License

This project is developed for educational and academic purposes.
