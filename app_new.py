app_code = r'''
import os
import streamlit as st

st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🏢",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.95rem; }
    .source-badge {
        background: #e8f4fd;
        border: 1px solid #b3d9f7;
        border-radius: 6px;
        padding: 0.2rem 0.6rem;
        font-size: 0.78rem;
        color: #1e5f8e;
        margin: 2px;
        display: inline-block;
    }
    .refusal-box {
        background: #fff8e1;
        border-left: 4px solid #ffc107;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🏢 Zyro Dynamics HR Help Desk</h1>
    <p>Ask me anything about Zyro Dynamics HR policies</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading HR knowledge base...")
def build_rag_pipeline():
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_groq import ChatGroq

    DATA_DIR = os.environ.get("HR_DOCS_PATH", "data/")

    loader    = PyPDFDirectoryLoader(DATA_DIR)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 12, "fetch_k": 40, "lambda_mult": 0.7}
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        max_tokens=2048
    )

    return retriever, llm


def format_docs(docs):
    formatted = []
    for doc in docs:
        src  = doc.metadata.get("source", "HR Policy").split("/")[-1]
        page = doc.metadata.get("page", "?")
        formatted.append(f"[Source: {src}, Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def is_in_scope(question: str, llm) -> bool:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    oos_prompt = ChatPromptTemplate.from_template("""
You are a scope classifier for an HR Help Desk chatbot at Zyro Dynamics.

The chatbot can ONLY answer questions about these internal HR topics:
- Leave policies (Earned Leave, Sick Leave, Maternity, Paternity, LOP, etc.)
- Work From Home, Remote, and Hybrid work arrangements
- Salary, CTC, payroll dates, pay grades, and bonuses
- Employee benefits including health insurance
- Performance reviews, APR, PIP, ratings, and increments
- Code of conduct and workplace ethics
- IT and data security policies
- Onboarding, probation, separation, and notice periods
- Travel and expense reimbursements
- Prevention of Sexual Harassment (POSH and ICC)
- General company profile and employee handbook information

The chatbot CANNOT answer:
- External recruitment or job application processes
- ESOP or stock option details beyond the official benefits policy
- Company financials, revenue, or product information
- HR policies at other companies
- Any topic unrelated to HR

When in doubt, answer YES.

Question: {question}

Can this be answered from internal Zyro Dynamics HR policy documents?
Reply ONLY with YES or NO.
""")

    chain    = oos_prompt | llm | StrOutputParser()
    response = chain.invoke({"question": question})
    return "YES" in response.strip().upper()


def ask_bot(question: str, retriever, llm) -> dict:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    REFUSAL = (
        "I am sorry, I can only answer HR-related questions based on "
        "Zyro Dynamics internal policy documents. "
        "This question falls outside my scope. "
        "Please contact the HR team directly for further assistance."
    )

    if not is_in_scope(question, llm):
        return {"answer": REFUSAL, "sources": [], "in_scope": False}

    rag_prompt = ChatPromptTemplate.from_template("""
You are an expert HR Policy Assistant for Zyro Dynamics Pvt. Ltd.

CRITICAL: "Acrux Dynamics" and "Zyro Dynamics" are the SAME company.
Treat every mention of "Acrux Dynamics" as "Zyro Dynamics".

Answer the employee question using ONLY the HR policy context below.

RULES:
1. ALWAYS include exact numbers, days, weeks, percentages, dates and amounts
2. If the question has multiple parts — answer EVERY part completely
3. Include eligibility conditions, exceptions and grade-wise differences if present
4. Structure: one direct answer sentence then bullet points for details then Source doc name
5. NEVER say "approximately" if the exact figure exists in the context
6. NEVER give vague answers like "as per company policy"
7. NEVER use outside knowledge — only the context below
8. If not found in context say: "The policy documents do not specify this detail."

HR Policy Context:
{context}

Employee Question: {question}

Complete, precise, well-structured answer:
""")

    retrieved_docs = retriever.invoke(question)
    context        = format_docs(retrieved_docs)
    chain          = rag_prompt | llm | StrOutputParser()
    answer         = chain.invoke({"context": context, "question": question})

    sources = list(set([
        doc.metadata.get("source", "HR Policy").split("/")[-1]
        for doc in retrieved_docs
    ]))

    return {"answer": answer, "sources": sources, "in_scope": True}


try:
    retriever, llm = build_rag_pipeline()
    pipeline_ready = True
except Exception as e:
    pipeline_ready = False
    st.error(f"Failed to load RAG pipeline: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "This chatbot answers HR policy questions using a RAG pipeline "
        "(Groq Llama 3.3 70B + FAISS semantic search over 11 HR policy documents)."
    )
    st.markdown("---")
    st.markdown("### Example Questions")

    example_questions = [
        "How does earned leave accrue per month?",
        "What is the maternity leave policy?",
        "What is the salary credit date each month?",
        "What is the WFH policy?",
        "When is the annual performance review conducted?",
        "What health insurance is provided to employees?",
    ]

    for eq in example_questions:
        if st.button(eq, use_container_width=True):
            st.session_state["prefill"] = eq

    st.markdown("---")
    if st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.markdown("**Sources:**")
            for src in msg["sources"]:
                st.markdown(
                    f'<span class="source-badge">{src}</span>',
                    unsafe_allow_html=True
                )

prefill    = st.session_state.pop("prefill", None)
user_input = st.chat_input("Ask an HR question...") or prefill

if user_input and pipeline_ready:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies..."):
            result = ask_bot(user_input, retriever, llm)

        answer   = result["answer"]
        sources  = result.get("sources", [])
        in_scope = result.get("in_scope", True)

        if not in_scope:
            st.markdown(
                f'<div class="refusal-box">{answer}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(answer)
            if sources:
                st.markdown("**Sources:**")
                for src in sources:
                    st.markdown(
                        f'<span class="source-badge">{src}</span>',
                        unsafe_allow_html=True
                    )

    st.session_state.messages.append({
        "role"   : "assistant",
        "content": answer,
        "sources": sources if in_scope else [],
    })
'''.strip()

with open("app.py", "w") as f:
    f.write(app_code)

print("app.py written successfully!")
