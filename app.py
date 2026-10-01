import streamlit as st
import ollama

st.set_page_config(
    page_title="Ollama Chatbot",
    page_icon="🤖"
)

st.title("🤖 Ollama Chatbot")
st.write("Ask anything!")

prompt = st.text_area(
    "Enter your prompt:",
    placeholder="Ask something..."
)

if st.button("Generate Response"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        try:
            response = ollama.chat(
                model="llama3.2",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            st.subheader("Response")
            st.write(response["message"]["content"])

        except Exception as e:
            st.error(
                "Could not connect to Ollama. "
                "Make sure Ollama is running and llama3.2 is installed."
            )
            st.code(str(e))