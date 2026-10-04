import streamlit as st
import chromadb
import ollama
import whisper
import sqlite3
import uuid
from datetime import datetime

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from PIL import Image

import tempfile
import hashlib
import os


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="CampusLens AI",
    page_icon="🎓",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")


embedding_model = load_embedding_model()


# --------------------------------------------------
# CHROMADB
# --------------------------------------------------

@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path="./chroma_db")


client = get_chroma_client()

collection = client.get_or_create_collection(
    name="campus_documents"
)


# --------------------------------------------------
# SQLITE CHAT HISTORY DATABASE
# --------------------------------------------------

CHAT_DB = "chat_history.db"


def get_chat_connection():
    return sqlite3.connect(
        CHAT_DB,
        check_same_thread=False
    )


def init_chat_db():

    conn = get_chat_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            chat_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def create_chat(chat_id, first_question):

    title = first_question.strip()

    if len(title) > 35:
        title = title[:35] + "..."

    if not title:
        title = "New Chat"

    now = datetime.now().isoformat()

    conn = get_chat_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR IGNORE INTO chats
        (chat_id, title, created_at, updated_at)
        VALUES (?, ?, ?, ?)
    """, (
        chat_id,
        title,
        now,
        now
    ))

    conn.commit()
    conn.close()


def save_message(chat_id, role, content):

    conn = get_chat_connection()
    cursor = conn.cursor()

    now = datetime.now().isoformat()

    cursor.execute("""
        INSERT INTO messages
        (chat_id, role, content, created_at)
        VALUES (?, ?, ?, ?)
    """, (
        chat_id,
        role,
        content,
        now
    ))

    cursor.execute("""
        UPDATE chats
        SET updated_at = ?
        WHERE chat_id = ?
    """, (
        now,
        chat_id
    ))

    conn.commit()
    conn.close()


def load_chat(chat_id):

    conn = get_chat_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT role, content
        FROM messages
        WHERE chat_id = ?
        ORDER BY id ASC
    """, (chat_id,))

    rows = cursor.fetchall()

    conn.close()

    return [
        {
            "role": role,
            "content": content
        }
        for role, content in rows
    ]


def get_all_chats():

    conn = get_chat_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT chat_id, title
        FROM chats
        ORDER BY updated_at DESC
    """)

    chats = cursor.fetchall()

    conn.close()

    return chats


def delete_chat(chat_id):

    conn = get_chat_connection()
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM messages WHERE chat_id = ?",
        (chat_id,)
    )

    cursor.execute(
        "DELETE FROM chats WHERE chat_id = ?",
        (chat_id,)
    )

    conn.commit()
    conn.close()


init_chat_db()


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = str(uuid.uuid4())


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def create_hash(data):
    return hashlib.sha256(data).hexdigest()


def split_text(text, chunk_size=700, overlap=100):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def add_to_database(text, source):

    chunks = split_text(text)

    if not chunks:
        return 0

    embeddings = embedding_model.encode(
        chunks
    ).tolist()

    ids = []
    metadatas = []

    for i, chunk in enumerate(chunks):

        chunk_id = hashlib.sha256(
            f"{source}-{i}-{chunk}".encode()
        ).hexdigest()

        ids.append(chunk_id)

        metadatas.append({
            "source": source,
            "chunk_number": i
        })

    try:

        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

        return len(chunks)

    except Exception as e:

        st.error(
            f"Database error: {e}"
        )

        return 0


# --------------------------------------------------
# PDF EXTRACTION
# --------------------------------------------------

def extract_pdf_text(uploaded_file):

    text = ""

    try:

        reader = PdfReader(uploaded_file)

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return text

    except Exception as e:

        st.error(
            f"PDF extraction failed: {e}"
        )

        return ""


# --------------------------------------------------
# AUDIO TRANSCRIPTION
# --------------------------------------------------

def transcribe_audio(uploaded_file):

    temp_path = None

    try:

        whisper_model = load_whisper_model()

        suffix = os.path.splitext(
            uploaded_file.name
        )[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            temp_path = temp_file.name

        result = whisper_model.transcribe(
            temp_path
        )

        return result["text"]

    except Exception as e:

        st.error(
            "Audio transcription failed. "
            "Make sure FFmpeg is installed."
        )

        st.error(str(e))

        return ""

    finally:

        if (
            temp_path
            and os.path.exists(temp_path)
        ):
            os.remove(temp_path)


# --------------------------------------------------
# MEMORY-AWARE SEARCH QUERY
# --------------------------------------------------

def build_search_query(question):

    previous_user_messages = []

    # Ignore the latest user message because
    # question is already passed separately.

    previous_messages = (
        st.session_state.messages[:-1]
        if st.session_state.messages
        else []
    )

    for message in previous_messages[-6:]:

        if message["role"] == "user":

            previous_user_messages.append(
                message["content"]
            )

    if previous_user_messages:

        previous_context = " ".join(
            previous_user_messages[-2:]
        )

        return (
            f"{previous_context} {question}"
        )

    return question


# --------------------------------------------------
# RETRIEVE FROM CHROMADB
# --------------------------------------------------

def retrieve_context(
    question,
    top_k=4
):

    if collection.count() == 0:

        return [], []

    search_query = build_search_query(
        question
    )

    question_embedding = (
        embedding_model.encode(
            [search_query]
        ).tolist()
    )

    results = collection.query(

        query_embeddings=question_embedding,

        n_results=min(
            top_k,
            collection.count()
        )
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    return documents, metadatas


# --------------------------------------------------
# GENERATE ANSWER
# --------------------------------------------------

def generate_answer(
    question,
    contexts,
    metadatas
):

    if not contexts:

        return (
            "I couldn't find relevant information "
            "in the CampusLens knowledge base."
        )

    context_text = ""

    for i, context in enumerate(contexts):

        source = metadatas[i].get(
            "source",
            "Unknown Source"
        )

        context_text += (
            f"\nSOURCE: {source}\n"
            f"{context}\n"
        )

    memory_text = ""

    for message in (
        st.session_state.messages[-8:]
    ):

        memory_text += (
            f"{message['role'].upper()}: "
            f"{message['content']}\n"
        )

    prompt = f"""
You are CampusLens, an AI assistant
for college information.

Answer using ONLY the retrieved
campus information given below.

Previous conversation may be used
only to understand follow-up questions.

If the requested information is not
available, say:

"I could not find this information in
the uploaded campus data."

Do not invent dates, deadlines,
rules, locations or requirements.

Keep the answer simple,
clear and student-friendly.

PREVIOUS CONVERSATION:
{memory_text}

RETRIEVED CAMPUS INFORMATION:
{context_text}

STUDENT QUESTION:
{question}

Give a clear and concise answer.
"""

    try:

        response = ollama.chat(

            model="llama3.2:1b",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response[
            "message"
        ]["content"]

    except Exception as e:

        return (
            "Could not connect to Ollama.\n\n"
            f"Error: {e}\n\n"
            "Make sure Ollama is running "
            "and llama3.2:1b is installed."
        )


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🎓 CampusLens AI")

st.caption(
    "Your intelligent assistant for "
    "college notices, circulars and announcements."
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("🎓 CampusLens")

    # -------------------------------
    # NEW CHAT
    # -------------------------------

    if st.button(
        "➕ New Chat",
        use_container_width=True
    ):

        st.session_state.current_chat_id = (
            str(uuid.uuid4())
        )

        st.session_state.messages = []

        st.rerun()

    st.divider()

    # -------------------------------
    # CHAT HISTORY
    # -------------------------------

    st.subheader("💬 Chat History")

    previous_chats = get_all_chats()

    if previous_chats:

        for chat_id, title in previous_chats:

            selected = (
                chat_id
                ==
                st.session_state.current_chat_id
            )

            button_text = (
                f"▶ {title}"
                if selected
                else title
            )

            if st.button(
                button_text,
                key=f"chat_{chat_id}",
                use_container_width=True
            ):

                st.session_state.current_chat_id = (
                    chat_id
                )

                st.session_state.messages = (
                    load_chat(chat_id)
                )

                st.rerun()

    else:

        st.caption(
            "No previous chats yet."
        )

    st.divider()

    # -------------------------------
    # DELETE CURRENT CHAT
    # -------------------------------

    if st.button(
        "🗑️ Delete Current Chat",
        use_container_width=True
    ):

        current_id = (
            st.session_state.current_chat_id
        )

        delete_chat(current_id)

        st.session_state.current_chat_id = (
            str(uuid.uuid4())
        )

        st.session_state.messages = []

        st.rerun()

    st.divider()

    # -------------------------------
    # KNOWLEDGE BASE
    # -------------------------------

    st.subheader("📚 Knowledge Base")

    st.metric(
        "Stored Chunks",
        collection.count()
    )

    if st.button(
        "⚠️ Clear Knowledge Base",
        use_container_width=True
    ):

        try:

            client.delete_collection(
                name="campus_documents"
            )

            st.session_state.processed_files = (
                set()
            )

            st.success(
                "Knowledge base cleared."
            )

            st.rerun()

        except Exception as e:

            st.error(e)


# --------------------------------------------------
# TABS
# --------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📄 PDF",
        "📝 Text Notice",
        "🎧 Audio",
        "🖼️ Image"
    ]
)


# --------------------------------------------------
# PDF TAB
# --------------------------------------------------

with tab1:

    st.subheader(
        "📄 Upload College PDF"
    )

    pdf_files = st.file_uploader(
        "Upload circulars, rules or notices",
        type=["pdf"],
        accept_multiple_files=True
    )

    if pdf_files:

        for pdf in pdf_files:

            file_bytes = pdf.getvalue()

            file_hash = create_hash(
                file_bytes
            )

            if (
                file_hash
                in st.session_state.processed_files
            ):

                st.info(
                    f"{pdf.name} already processed."
                )

                continue

            if st.button(
                f"Process {pdf.name}",
                key=f"pdf_{file_hash}"
            ):

                with st.spinner(
                    f"Reading {pdf.name}..."
                ):

                    text = extract_pdf_text(
                        pdf
                    )

                if text.strip():

                    chunks = add_to_database(
                        text,
                        pdf.name
                    )

                    st.session_state.processed_files.add(
                        file_hash
                    )

                    st.success(
                        f"{pdf.name} processed "
                        f"successfully. "
                        f"{chunks} chunks stored."
                    )

                else:

                    st.warning(
                        "No readable text found "
                        "in this PDF."
                    )


# --------------------------------------------------
# TEXT NOTICE TAB
# --------------------------------------------------

with tab2:

    st.subheader(
        "📝 Add Text Notice"
    )

    notice_title = st.text_input(
        "Notice title",
        placeholder=(
            "Example: Hackathon Announcement"
        )
    )

    notice_text = st.text_area(
        "Paste the notice",
        height=180,
        placeholder="""
Example:

Hackathon registrations close on October 12.
Each team must contain 3 to 4 members.
The event will be conducted in the college auditorium.
"""
    )

    if st.button(
        "➕ Add Notice"
    ):

        if notice_text.strip():

            source = (
                notice_title.strip()
                if notice_title.strip()
                else "Text Notice"
            )

            chunks = add_to_database(
                notice_text,
                source
            )

            st.success(
                f"Notice added successfully. "
                f"{chunks} chunks stored."
            )

        else:

            st.warning(
                "Please enter some notice text."
            )


# --------------------------------------------------
# AUDIO TAB
# --------------------------------------------------

with tab3:

    st.subheader(
        "🎧 Upload Audio Announcement"
    )

    audio_file = st.file_uploader(
        "Upload faculty announcement",
        type=[
            "mp3",
            "wav",
            "m4a",
            "opus"
        ],
        key="audio_uploader"
    )

    if audio_file:

        st.audio(audio_file)

        audio_bytes = (
            audio_file.getvalue()
        )

        audio_hash = create_hash(
            audio_bytes
        )

        if (
            audio_hash
            in st.session_state.processed_files
        ):

            st.info(
                "This audio file "
                "is already processed."
            )

        else:

            if st.button(
                "🎧 Transcribe and Add",
                key="audio_process"
            ):

                with st.spinner(
                    "Whisper is transcribing "
                    "the audio..."
                ):

                    transcript = (
                        transcribe_audio(
                            audio_file
                        )
                    )

                if transcript.strip():

                    st.subheader(
                        "Transcript"
                    )

                    st.write(
                        transcript
                    )

                    chunks = add_to_database(
                        transcript,
                        audio_file.name
                    )

                    st.session_state.processed_files.add(
                        audio_hash
                    )

                    st.success(
                        "Audio processed successfully. "
                        f"{chunks} chunks stored."
                    )


# --------------------------------------------------
# IMAGE TAB
# --------------------------------------------------

with tab4:

    st.subheader(
        "🖼️ Upload Campus Image"
    )

    image_file = st.file_uploader(
        "Upload poster or campus image",
        type=[
            "png",
            "jpg",
            "jpeg"
        ],
        key="image_uploader"
    )

    if image_file:

        image = Image.open(
            image_file
        )

        st.image(
            image,
            caption=image_file.name,
            use_container_width=True
        )

        image_description = st.text_area(
            "Add information about this image",
            placeholder=(
                "Example: Hackathon registration "
                "closes on October 10. "
                "Team size is 3-4 students."
            ),
            key="image_description"
        )

        if st.button(
            "➕ Add Image Information",
            key="add_image_info"
        ):

            if image_description.strip():

                chunks = add_to_database(
                    image_description,
                    image_file.name
                )

                st.success(
                    "Image information added "
                    "successfully. "
                    f"{chunks} chunks stored."
                )

            else:

                st.warning(
                    "Please enter information "
                    "about the image first."
                )


# --------------------------------------------------
# DIVIDER
# --------------------------------------------------

st.divider()


# --------------------------------------------------
# CHAT
# --------------------------------------------------

st.header("💬 Ask CampusLens")


# DISPLAY CURRENT CHAT

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# --------------------------------------------------
# USER QUESTION
# --------------------------------------------------

question = st.chat_input(
    "Ask about campus notices, "
    "deadlines, rules..."
)


if question:

    current_chat_id = (
        st.session_state.current_chat_id
    )

    # Create persistent chat on first question

    if not st.session_state.messages:

        create_chat(
            current_chat_id,
            question
        )

    # ----------------------------------------------
    # STORE USER MESSAGE
    # ----------------------------------------------

    user_message = {
        "role": "user",
        "content": question
    }

    st.session_state.messages.append(
        user_message
    )

    save_message(
        current_chat_id,
        "user",
        question
    )

    with st.chat_message("user"):

        st.markdown(question)


    # ----------------------------------------------
    # RETRIEVE
    # ----------------------------------------------

    with st.spinner(
        "Searching campus information..."
    ):

        contexts, metadatas = (
            retrieve_context(
                question
            )
        )


    # ----------------------------------------------
    # GENERATE
    # ----------------------------------------------

    with st.spinner(
        "CampusLens is thinking..."
    ):

        answer = generate_answer(
            question,
            contexts,
            metadatas
        )


    # ----------------------------------------------
    # STORE ASSISTANT RESPONSE
    # ----------------------------------------------

    assistant_message = {
        "role": "assistant",
        "content": answer
    }

    st.session_state.messages.append(
        assistant_message
    )

    save_message(
        current_chat_id,
        "assistant",
        answer
    )


    # ----------------------------------------------
    # DISPLAY RESPONSE
    # ----------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        st.markdown(answer)

        if contexts:

            st.markdown(
                "### 📚 Sources"
            )

            sources = []

            for metadata in metadatas:

                source = metadata.get(
                    "source",
                    "Unknown"
                )

                if source not in sources:
                    sources.append(
                        source
                    )

            for source in sources:

                st.write(
                    f"• {source}"
                )

            with st.expander(
                "🔍 View Retrieved Context"
            ):

                for i, context in enumerate(
                    contexts
                ):

                    source = (
                        metadatas[i].get(
                            "source",
                            "Unknown"
                        )
                    )

                    st.markdown(
                        f"**Source: {source}**"
                    )

                    st.write(
                        context
                    )

                    st.divider()