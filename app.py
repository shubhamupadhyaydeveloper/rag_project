# app.py
import streamlit as st
from answer import answer_question

st.set_page_config(
    page_title="Email Help Assistant",
    page_icon="🛡️",
    layout="centered"
)

st.title("Email Help Assistant")
st.caption("Ask anything about your emails!")

# chat history initialize karo
if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = []

# purane messages dikhao
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📄 Sources"):
                for src in message["sources"]:
                    st.caption(f"• {src}")

# user input
if prompt := st.chat_input("Ask about Emails..."):
    # user message dikhao
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # answer generate karo
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, docs = answer_question(prompt, st.session_state.history)

        st.markdown(answer)

        # sources dikhao
        sources = list(set(d.metadata.get("source", "") for d in docs))
        with st.expander("📄 Sources"):
            for src in sources:
                st.caption(f"• {src}")

    # history update karo
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
    st.session_state.history.append({"role": "user", "content": prompt})
    st.session_state.history.append({"role": "assistant", "content": answer})