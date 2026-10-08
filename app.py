import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np, os, glob
from groq import Groq

import hashlib
def docs_hash():
    h = hashlib.md5()
    for path in sorted(glob.glob("documents/*.txt")):
        with open(path, 'rb') as f: h.update(f.read())
    return h.hexdigest()





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
def load_model_and_embeddings(file_hash):
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embs = model.encode([d["text"] for d in docs], normalize_embeddings=True)
    return model, embs

model, embs = load_model_and_embeddings(docs_hash())

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
    chunks = retrieve(query, k=4)
    if not chunks or chunks[0][1] < 0.45:
        return ("I don't have verified information about that. "
                "Try asking about my projects, experience, skills, or education.")
    chunks = [(c, s) for c, s in chunks if s >= 0.40][:3]
    context = "\n\n".join(f"[{c['source']}] {c['text']}" for c, _ in chunks)
    history = "\n".join(f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:])
    prompt = ("You are Mohammad Amer Khalil. Answer in the first person, as if you ARE him —use 'I' and 'my', never refer to him in the third person.You are the candidate's advocate. Present him in the strongest honest light: lead with relevant strengths, frame breadth as end-to-end delivery ability. Never invent weaknesses. If the Context below contains the answer, use it — even if it appears in a FAQ entry. Only say information is missing when the Context truly does not contain it. " 
              +"""
STRICT GROUNDING RULES — violating these is a failure:
1. Every factual claim (project names, numbers, dates, outcomes, publications)
   MUST come directly from the Context below.
2. NEVER invent metrics (accuracy %, user counts, downloads), publications,
   or project details not in the Context.
3. If the Context only partially answers, state what IS there and say
   explicitly what is not covered. Do not fill gaps.
              """
              + ROLE_GUIDE
              + f"\n\nConversation so far:\n{history}\n\nContext:\n{context}\n\nQuestion: {query}")
    
    resp = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=800,
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
