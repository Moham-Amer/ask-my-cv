import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np, os, glob
from groq import Groq

if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]
else:
    api_key = os.environ.get("GROQ_API_KEY", "")

client = Groq(api_key=api_key)
docs = []
for path in sorted(glob.glob("documents/*.txt")):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    for para in text.split("\n\n"):
        para = para.strip()
        if len(para) > 40:
            docs.append({"text": para, "source": os.path.basename(path).replace(".txt", "")})


# @st.cache_resource so your app doesn't re-download the model on every click
@st.cache_resource
def load_model_and_embeddings():
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embs = model.encode([d["text"] for d in docs], normalize_embeddings=True)
    return model, embs

model, embs = load_model_and_embeddings()

def retrieve(query, k=3):
    q = model.encode([query], normalize_embeddings=True)[0]
    scores = embs @ q
    idx = np.argsort(scores)[::-1][:k]
    return [(docs[i], float(scores[i])) for i in idx]

ROLE_FRAMING = {
    "AI Engineer": (
        "The asker is hiring for an AI Engineer role. Lead with AI/ML experience "
        "(research, Arabic NLP, deep learning, transformers, LLM APIs). Mention software "
        "projects only briefly, as evidence that he ships real products around AI work."
    ),
    "Frontend Developer": (
        "The asker is hiring for a Frontend Developer role. Lead with React experience "
        "(production apps, component architecture, state management, testing). Mention AI/ML "
        "only briefly, as differentiating depth."
    ),
    "Mobile Developer": (
        "The asker is hiring for a Mobile Developer role. Lead with Flutter experience "
        "(shipped apps, Bloc/Cubit, clean architecture, bilingual UI). Mention other areas "
        "only briefly."
    ),
    "General / Other": (
        "Give a balanced answer across his background: AI research first (his differentiator), "
        "then shipped software (web, mobile, backend) as proof he delivers end-to-end."
    ),
}

st.set_page_config(page_title="AskMyCV", page_icon="📄")
st.title("AskMyCV — اسأل عن مؤهلاتي")
st.caption("Ask anything about my background (Arabic or English). Every answer comes from my CVs, with sources shown.")

ROLE_GUIDE = """If the user mentions a hiring role anywhere in the conversation
(e.g. "I'm hiring for an AI engineer"), frame the answer for that role:
- AI Engineer: lead with AI/ML (research, Arabic NLP, deep learning, LLM APIs); shipped apps only as proof he delivers.
- Frontend Developer: lead with React (production apps, state management, testing); AI briefly as differentiating depth.
- Mobile Developer: lead with Flutter (Bloc/Cubit, clean architecture, bilingual UI); the rest briefly.
If no role is mentioned: balanced answer, AI research first, then shipped software."""

def answer(query):
    chunks = retrieve(query)
    context = "\n\n".join(f"[{c['source']}] {c['text']}" for c, _ in chunks)
    history = "\n".join(f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:])
    prompt = ("You are the candidate's advocate... " + ROLE_GUIDE
              + f"\n\nConversation so far:\n{history}\n\nContext:\n{context}\n\nQuestion: {query}")
    
    resp = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2, max_tokens=700,
    )
    ans = resp.choices[0].message.content
    refs = "\n".join(f"- {c['source']} (match {s:.2f})" for c, s in chunks)
    return ans + f"\n\n**Sources:**\n{refs}"


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_query = st.chat_input("Ask a question about my CV...")
query_to_process = user_query 

if query_to_process:
    st.session_state.messages.append({"role": "user", "content": query_to_process})
    with st.chat_message("user"):
        st.markdown(query_to_process)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            bot_response = answer(query_to_process)
            st.markdown(bot_response)
            
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
