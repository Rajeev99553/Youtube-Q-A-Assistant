# import re
# import streamlit as st
# from dotenv import load_dotenv

# # LangChain + HuggingFace imports
# from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
# from langchain_core.output_parsers import StrOutputParser

# # ------------------------
# # Load API Keys
# # ------------------------
# load_dotenv()

# # ------------------------
# # Helper function: Extract video ID
# # ------------------------
# def extract_video_id(url: str) -> str:
#     """
#     Extract YouTube video ID from URL.
#     Supports standard, short, and embed formats.
#     """
#     video_id_match = re.search(r"(?:v=|youtu\.be/|embed/)([a-zA-Z0-9_-]{11})", url)
#     if video_id_match:
#         return video_id_match.group(1)
#     else:
#         return None

# # ------------------------
# # Transcript Fetching
# # ------------------------
# def fetch_transcript(video_id: str, languages=["en"]) -> str:
#     ytt = YouTubeTranscriptApi()
#     try:
#         fetched = ytt.fetch(video_id, languages=languages)
#         transcript = " ".join(snippet.text for snippet in fetched)
#         return transcript
#     except TranscriptsDisabled:
#         st.error("❌ Captions are disabled for this video.")
#     except NoTranscriptFound:
#         st.error("❌ No transcript found in the requested languages.")
#     except VideoUnavailable:
#         st.error("❌ The video is unavailable.")
#     except Exception as e:
#         st.error(f"⚠️ Unexpected error: {e}")
#     return None

# # ------------------------
# # LangChain Setup
# # ------------------------
# def build_qa_chain(vector_store):
#     # Retriever
#     retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

#     # Prompt
#     prompt = PromptTemplate(
#         template="""
# You are a helpful assistant.
# Answer ONLY from the provided transcript context.
# If the context is insufficient, just say you don't know.

# {context}
# Question: {question}
# """,
#         input_variables=['context', 'question']
#     )

#     # Model
#     llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct")
#     model = ChatHuggingFace(llm=llm)

#     # Pipeline
#     def format_docs(retrieved_docs):
#         return "\n\n".join(doc.page_content for doc in retrieved_docs)

#     parallel_chain = RunnableParallel({
#         "context": retriever | RunnableLambda(format_docs),
#         "question": RunnablePassthrough()
#     })

#     parser = StrOutputParser()
#     main_chain = parallel_chain | prompt | model | parser
#     return main_chain

# # ------------------------
# # Main Streamlit App
# # ------------------------
# def main():
#     st.set_page_config(page_title="YouTube Q&A Assistant", layout="centered")
#     st.title("🎬 YouTube Q&A Assistant")

#     url = st.text_input("Paste YouTube Video URL here:")

#     if st.button("Process Video"):
#         if not url:
#             st.warning("⚠️ Please enter a YouTube URL.")
#             return

#         video_id = extract_video_id(url)
#         if not video_id:
#             st.error("Invalid YouTube URL. Could not extract video ID.")
#             return

#         with st.spinner("Loading Your Video... Please wait"):
#             transcript = fetch_transcript(video_id)

#             if transcript:
#                 # Split transcript
#                 splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#                 chunks = splitter.create_documents([transcript])

#                 # Create embeddings + vector store
#                 embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
#                 vector_store = FAISS.from_documents(chunks, embeddings)

#                 # Save to session
#                 st.session_state.vector_store = vector_store
#                 st.session_state.transcript = transcript

#                 st.success("Transcript processed and vector store created!")
#                 # st.write("Here’s a preview of the transcript:")
#                 # st.write(transcript[:800] + "..." if len(transcript) > 800 else transcript)

#     # Q&A Section
#     if "vector_store" in st.session_state:
#         st.subheader("Ask Questions about the Video")
#         question = st.text_input("Your question:")

#         if st.button("Get Answer"):
#             if question.strip():
#                 qa_chain = build_qa_chain(st.session_state.vector_store)
#                 with st.spinner("Thinking..."):
#                     answer = qa_chain.invoke(question)
#                 st.success("Answer")
#                 st.write(answer)
#             else:
#                 st.warning("⚠️ Please enter a question.")

# if __name__ == "__main__":
#     main()

#New Code

# app.py
import re
import io
import streamlit as st
from dotenv import load_dotenv
import warnings

# suppress some noisy future warnings (optional)
warnings.filterwarnings("ignore", category=FutureWarning)

# YouTube transcript API
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

# LangChain pieces
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ------------------------
# Utilities: robust transcript fetcher
# ------------------------
def fetch_transcript_robust(video_id: str, languages=("en",)):
    """
    Try multiple API variants for youtube-transcript-api to be robust across versions.
    Returns plain text transcript or None (and writes an st.error).
    """
    last_exc = None

    # 1) try static/class method get_transcript
    try:
        if hasattr(YouTubeTranscriptApi, "get_transcript"):
            tl = YouTubeTranscriptApi.get_transcript(video_id, languages=list(languages))
            txt = _normalize_transcript_list_to_text(tl)
            return txt
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        # surface known exceptions immediately
        raise
    except Exception as e:
        last_exc = e

    # 2) try instance.fetch (some versions expose fetch as instance method)
    try:
        try:
            ytt = YouTubeTranscriptApi()
        except TypeError:
            # some versions might not be instantiable; skip
            ytt = None

        if ytt is not None and hasattr(ytt, "fetch"):
            fetched = ytt.fetch(video_id, languages=list(languages))
            txt = _normalize_transcript_list_to_text(fetched)
            return txt
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        raise
    except Exception as e:
        last_exc = e

    # 3) try get_transcripts / get_transcripts plural
    try:
        if hasattr(YouTubeTranscriptApi, "get_transcripts"):
            resp = YouTubeTranscriptApi.get_transcripts([video_id], languages=list(languages))
            # resp can have multiple shapes; try to find the actual list of pieces
            transcript_list = None
            if isinstance(resp, dict) and video_id in resp:
                transcript_list = resp[video_id]
            elif isinstance(resp, list) and len(resp) > 0:
                # try first item or look for mapping with video_id
                first = resp[0]
                if isinstance(first, dict) and video_id in first:
                    transcript_list = first[video_id]
                else:
                    transcript_list = resp
            elif isinstance(resp, dict):
                # flatten values
                candidate = []
                for v in resp.values():
                    if isinstance(v, list):
                        candidate.extend(v)
                transcript_list = candidate
            else:
                transcript_list = resp

            txt = _normalize_transcript_list_to_text(transcript_list)
            return txt
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        raise
    except Exception as e:
        last_exc = e

    # If we reach here, nothing worked
    st.error(f"Could not fetch transcript. Last error: {last_exc}")
    return None


def _normalize_transcript_list_to_text(transcript_list):
    """
    Normalize many possible transcript return shapes into a single plain string.
    Accepts lists of dicts like [{'text':...}, ...] or objects with .text attributes.
    """
    if transcript_list is None:
        return ""
    # If it's a dict mapping language -> list, try to pick first list
    if isinstance(transcript_list, dict):
        # find first list-like value
        for v in transcript_list.values():
            if isinstance(v, list):
                transcript_list = v
                break

    # If iterable of dicts or objects -> join
    out_texts = []
    try:
        for chunk in transcript_list:
            if isinstance(chunk, dict):
                out_texts.append(chunk.get("text", ""))
            else:
                # try attributes (some versions use small objects)
                text = getattr(chunk, "text", None)
                if text is None:
                    # fallback to str(chunk)
                    text = str(chunk)
                out_texts.append(text)
    except Exception:
        # final fallback: turn the whole thing into a string
        return str(transcript_list)

    return " ".join(t for t in out_texts if t)


# ------------------------
# LangChain: QA + Summarization builders
# ------------------------
def build_qa_chain(vector_store, model):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    prompt = PromptTemplate(
        template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, say you don't know.

{context}
Question: {question}
""",
        input_variables=["context", "question"],
    )

    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )

    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | model | parser
    return main_chain


def build_summary_chain_for_transcript(model, transcript_text):
    """
    Builds a runnable that injects the full transcript_text as 'context' and expects
    a 'question' input (we pass "Summarize the entire video..." as the question).
    """
    prompt = PromptTemplate(
        template="""
You are a helpful assistant.
Summarize the transcript below in detail. Produce:
1) A short (2-3 sentence) summary.
2) Key bullet points (6-12 bullets) with timestamps if present in the transcript.
3) A short conclusion or takeaway.

{context}
Question: {question}
""",
        input_variables=["context", "question"],
    )

    def return_context(_: str):
        return transcript_text

    parallel_chain = RunnableParallel(
        {
            "context": RunnableLambda(return_context),
            "question": RunnablePassthrough(),
        }
    )

    parser = StrOutputParser()
    chain = parallel_chain | prompt | model | parser
    return chain


def summarize_transcript_hierarchical(model, transcript_text, max_chars_single_call=30000):
    """
    If transcript is small enough, send full transcript to LLM once.
    Otherwise, create chunk summaries then combine (hierarchical summarization).
    """
    if not transcript_text:
        return "No transcript available to summarize."

    if len(transcript_text) <= max_chars_single_call:
        chain = build_summary_chain_for_transcript(model, transcript_text)
        return chain.invoke("Summarize the entire video in detail.")
    # else hierarchical
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    docs = splitter.create_documents([transcript_text])
    chunk_summaries = []
    for i, d in enumerate(docs, start=1):
        # summarise each chunk briefly
        cchain = build_summary_chain_for_transcript(model, d.page_content)
        s = cchain.invoke("Summarize this chunk in 2-4 sentences (concise).")
        chunk_summaries.append(f"Chunk {i} summary:\n{s}")

    combined = "\n\n".join(chunk_summaries)
    # If combined still too big, reduce recursively
    if len(combined) > max_chars_single_call:
        return summarize_transcript_hierarchical(model, combined, max_chars_single_call)
    # Final combined refinement
    final_chain = build_summary_chain_for_transcript(model, combined)
    return final_chain.invoke("Combine and refine the chunk summaries into a single detailed summary.")


# ------------------------
# Helper: create downloadable DOCX and PDF bytes (optional)
# ------------------------
def create_docx_bytes(text: str):
    try:
        from docx import Document
    except Exception:
        return None
    doc = Document()
    for para in text.split("\n\n"):
        doc.add_paragraph(para)
    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio.getvalue()


def create_pdf_bytes(text: str):
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import simpleSplit
    except Exception:
        return None
    bio = io.BytesIO()
    width, height = letter
    c = canvas.Canvas(bio, pagesize=letter)
    lines = simpleSplit(text, "Helvetica", 10, width - 72)
    y = height - 72
    for line in lines:
        if y < 72:
            c.showPage()
            y = height - 72
        c.drawString(36, y, line)
        y -= 12
    c.save()
    bio.seek(0)
    return bio.getvalue()


# ------------------------
# Streamlit App
# ------------------------
def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/|embed/)([A-Za-z0-9_-]{11})", url)
    return match.group(1) if match else None


def main():
    st.set_page_config(page_title="YouTube Q&A Assistant", layout="wide")

    # CSS
    st.markdown(
        """
    <style>
    .big-title { font-size:36px !important; font-weight:700; color:#FF4B4B; text-align:center; }
    .subtitle { font-size:18px !important; color:#333333; text-align:center; margin-bottom:12px; }
    .question-box { background:#f9f9f9; padding:10px 15px; border-radius:10px; margin:8px 0; }
    .answer-box { background:#e6f4ff; padding:10px 15px; border-radius:10px; margin:8px 0; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="big-title">🎬 YouTube Q&A Assistant</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Paste a YouTube URL, process the transcript, ask questions, or summarize the entire video.</p>',
        unsafe_allow_html=True,
    )

    # Input
    url = st.text_input("🔗 Paste YouTube Video URL here:")

    if st.button("🚀 Process Video", use_container_width=True):
        if not url:
            st.warning("Please enter a YouTube URL.")
            return

        video_id = extract_video_id(url)
        if not video_id:
            st.error("Invalid YouTube URL. Could not extract video ID.")
            return

        with st.spinner("⏳ Fetching transcript..."):
            try:
                transcript = fetch_transcript_robust(video_id, languages=("en",))
            except TranscriptsDisabled:
                st.error("Captions are disabled for this video.")
                return
            except NoTranscriptFound:
                st.error("No transcript found in the requested languages.")
                return
            except VideoUnavailable:
                st.error("The video is unavailable.")
                return
            except Exception as e:
                st.error(f"Unexpected error while fetching transcript: {e}")
                return

            if not transcript:
                st.error("No transcript could be retrieved.")
                return

            # Split, embed, vectorstore
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.create_documents([transcript])

            with st.spinner("⏳ Creating embeddings & vector store..."):
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vector_store = FAISS.from_documents(chunks, embeddings)

            # Save useful objects in session_state
            st.session_state.vector_store = vector_store
            st.session_state.transcript = transcript
            st.session_state.qa_history = []

            st.success("✅ Transcript processed and vector store is ready.")
            st.video(f"https://www.youtube.com/watch?v={video_id}")

            with st.expander("📜 Transcript preview"):
                st.write(transcript[:2000] + "..." if len(transcript) > 2000 else transcript)

            # Download transcript (txt, docx, pdf)
            st.download_button(
                "⬇️ Download Transcript (.txt)",
                transcript,
                file_name="transcript.txt",
                mime="text/plain",
                use_container_width=True,
            )

            docx_bytes = create_docx_bytes(transcript)
            if docx_bytes:
                st.download_button(
                    "⬇️ Download Transcript (.docx)",
                    data=docx_bytes,
                    file_name="transcript.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                )
            else:
                st.info("Install `python-docx` to enable .docx download (pip install python-docx).")

            pdf_bytes = create_pdf_bytes(transcript)
            if pdf_bytes:
                st.download_button(
                    "⬇️ Download Transcript (.pdf)",
                    data=pdf_bytes,
                    file_name="transcript.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            else:
                st.info("Install `reportlab` to enable PDF download (pip install reportlab).")

    # Q&A and Summary UI (requires vector store)
    if "vector_store" in st.session_state:
        # instantiate / cache model in session to avoid re-creating endpoint repeatedly
        if "llm_model" not in st.session_state:
            st.session_state.llm_model = ChatHuggingFace(
                llm=HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct")
            )
        model = st.session_state.llm_model
        vector_store = st.session_state.vector_store
        transcript = st.session_state.transcript

        st.subheader("💬 Ask Questions or Summarize")

        cols = st.columns([3, 1])
        question = cols[0].text_input("❓ Your question:")
        # buttons in second column
        ask_btn = cols[1].button("🤖 Ask")
        summarize_btn = cols[1].button("📝 Summarize Entire Video")

        if ask_btn and question.strip():
            qa_chain = build_qa_chain(vector_store, model)
            with st.spinner("🤔 Thinking..."):
                try:
                    answer = qa_chain.invoke(question)
                except Exception as e:
                    st.error(f"Model error: {e}")
                    answer = "Model call failed."
            st.session_state.qa_history.append((question, answer))

        if summarize_btn:
            with st.spinner("📚 Summarizing entire transcript (may take a bit)..."):
                try:
                    summary = summarize_transcript_hierarchical(model, transcript)
                except Exception as e:
                    st.error(f"Summarization failed: {e}")
                    summary = "Summarization failed."
            st.session_state.qa_history.append(("Summarize the entire video", summary))

        # Show conversation history and provide downloads
        if st.session_state.qa_history:
            st.markdown("### 📝 Conversation / Summary History")
            conversation_text = ""
            for q, a in st.session_state.qa_history:
                st.markdown(f'<div class="question-box"><b>Q:</b> {q}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="answer-box"><b>A:</b> {a}</div>', unsafe_allow_html=True)
                conversation_text += f"Q: {q}\nA: {a}\n\n"

            st.download_button(
                "⬇️ Download Conversation / Summaries (.txt)",
                conversation_text,
                file_name="qa_summary.txt",
                mime="text/plain",
                use_container_width=True,
            )

            docx_bytes = create_docx_bytes(conversation_text)
            if docx_bytes:
                st.download_button(
                    "⬇️ Download Conversation / Summaries (.docx)",
                    data=docx_bytes,
                    file_name="qa_summary.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                )

            pdf_bytes = create_pdf_bytes(conversation_text)
            if pdf_bytes:
                st.download_button(
                    "⬇️ Download Conversation / Summaries (.pdf)",
                    data=pdf_bytes,
                    file_name="qa_summary.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()
