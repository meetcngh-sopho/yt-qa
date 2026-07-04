from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import chromadb, re, os, json

# -------------------------------------------------------
# YouTube Q&A — FastAPI + RAG
#
# Deployment modes (auto-detected via env vars):
#   Local  → Ollama for chat + embeddings, persistent ChromaDB
#   Railway → Groq for chat, sentence-transformers for embeddings, in-memory ChromaDB
#
# Environment variables (set in Railway dashboard):
#   GROQ_API_KEY  → Groq API key (required for Railway)
#   DEPLOY_MODE   → "railway" to switch to cloud mode (default: local)
# -------------------------------------------------------

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR     = BASE_DIR if os.name == "nt" else "/tmp"
STATIC_DIR   = os.path.join(BASE_DIR, "static")
DEPLOY_MODE  = os.environ.get("DEPLOY_MODE", "local")  # "local" or "railway"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ── Model config based on deploy mode ──────────────────
if DEPLOY_MODE == "railway":
    MODEL  = "llama-3.3-70b-versatile"
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
    chroma = chromadb.Client()  # in-memory on Railway
    CHROMA_DIR    = None
    METADATA_FILE = None
    print("[Config] Mode: Railway | Model: Groq llama-3.3-70b | ChromaDB: in-memory")
else:
    MODEL          = "llama3.2:3b"
    OLLAMA_BASE_URL = "http://localhost:11434" if os.name == "nt" else "http://Jagmeet-singh.local:11434"
    client         = OpenAI(base_url=f"{OLLAMA_BASE_URL}/v1", api_key="ollama")
    CHROMA_DIR     = os.path.join(BASE_DIR, "chroma_db")
    METADATA_FILE  = os.path.join(BASE_DIR, "videos_metadata.json")
    chroma         = chromadb.PersistentClient(path=CHROMA_DIR)
    print(f"[Config] Mode: Local | Model: Ollama {MODEL} | ChromaDB: persistent")

loaded_videos        = {}
conversation_history = {}  # video_id → list of {role, content}


def save_metadata():
    if not METADATA_FILE:
        return  # skip on Railway — no disk persistence
    with open(METADATA_FILE, "w") as f:
        json.dump(loaded_videos, f, indent=2)
    print(f"[Persist] Metadata saved → {METADATA_FILE}")


def load_metadata():
    global loaded_videos
    if not METADATA_FILE:
        return  # skip on Railway — start fresh
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE) as f:
            loaded_videos = json.load(f)
        print(f"[Persist] Loaded {len(loaded_videos)} videos from metadata file")
    else:
        print("[Persist] No metadata file found — starting fresh")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # on startup — reload saved metadata
    load_metadata()
    yield
    # on shutdown — save metadata
    save_metadata()


app = FastAPI(title="YouTube Q&A", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class LoadRequest(BaseModel):
    url: str


class AskRequest(BaseModel):
    video_id: str
    question: str
    k: int = 4


# -------------------------------------------------------
# Utilities
# -------------------------------------------------------
def get_video_id(url: str) -> str:
    for p in [r"v=([a-zA-Z0-9_-]{11})", r"youtu\.be/([a-zA-Z0-9_-]{11})", r"shorts/([a-zA-Z0-9_-]{11})"]:
        m = re.search(p, url)
        if m:
            return m.group(1)
    raise ValueError(f"Cannot extract video ID from: {url}")


def clean_url(url: str) -> str:
    # strip playlist params — keep only the video URL
    video_id = get_video_id(url)
    return f"https://www.youtube.com/watch?v={video_id}"


def fetch_transcript_api(video_id: str):
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        # new API (v1.x): fetch() takes video_id directly, list() returns available transcripts
        transcript_list = YouTubeTranscriptApi.list(video_id=video_id)
        # try English first, then fall back to first available
        transcript = None
        for t in transcript_list:
            if t.language_code in ["en", "en-US", "en-GB"]:
                transcript = t
                break
        if transcript is None:
            transcript = next(iter(transcript_list))
        data = transcript.fetch()
        return " ".join([s.text for s in data]), "transcript_api"
    except Exception as e:
        print(f"[transcript_api] {e}")
        return None, None


def fetch_transcript_whisper(url: str, video_id: str):
    try:
        import yt_dlp
        from faster_whisper import WhisperModel
        audio = os.path.join(TEMP_DIR, f"yt_{video_id}.mp3")
        # find ffmpeg — check common locations including winget install path
        ffmpeg_locations = [
            r"C:\Users\meetc\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe",
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            "ffmpeg",  # fallback — use PATH
        ]
        ffmpeg_path = next((p for p in ffmpeg_locations if os.path.exists(p) or p == "ffmpeg"), "ffmpeg")

        opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(TEMP_DIR, f"yt_{video_id}.%(ext)s"),
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "96"}],
            "ffmpeg_location": os.path.dirname(ffmpeg_path),
            "quiet": True,
            "noplaylist": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        model = WhisperModel("small", device="cpu", compute_type="int8")
        segs, _ = model.transcribe(audio, beam_size=5)
        text = " ".join([s.text for s in segs])
        if os.path.exists(audio):
            os.remove(audio)
        return text, "whisper"
    except Exception as e:
        print(f"[Whisper] {e}")
        return None, None


def get_title(video_id: str) -> str:
    try:
        import yt_dlp
        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            return info.get("title", video_id)
    except Exception:
        return video_id


def chunk_text(text: str, size: int = 500, overlap: int = 50):
    words, chunks, i = text.split(), [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        i += size - overlap
    return chunks


def embed(text: str):
    if DEPLOY_MODE == "railway":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(text[:2000]).tolist()
    else:
        r = client.embeddings.create(model="nomic-embed-text", input=text[:2000])
        return r.data[0].embedding


def embed_chunks_parallel(chunks: list, max_workers: int = 4) -> list:
    # embed all chunks simultaneously instead of one by one
    results = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(embed, c): i for i, c in enumerate(chunks)}
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
    return results


# -------------------------------------------------------
# Routes
# -------------------------------------------------------
@app.get("/ui")
def ui():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/")
def root():
    return {
        "app":     "YouTube Q&A",
        "version": "1.0",
        "docs":    "http://localhost:8000/docs",
        "loaded_videos": len(loaded_videos),
        "endpoints": [
            {"method": "POST",   "path": "/load",               "description": "Load a YouTube video by URL — fetches transcript, chunks, embeds, stores"},
            {"method": "POST",   "path": "/ask",                "description": "Ask a question — RAG + conversation memory + streaming answer"},
            {"method": "GET",    "path": "/videos",             "description": "List all loaded videos with title, chunks, method"},
            {"method": "GET",    "path": "/status/{video_id}",  "description": "Check if a specific video is loaded"},
            {"method": "GET",    "path": "/content/{video_id}", "description": "Get full transcript and all chunks for a video"},
            {"method": "DELETE", "path": "/clear/{video_id}",   "description": "Clear conversation history for a video"},
            {"method": "GET",    "path": "/ui",                 "description": "Web UI"},
        ]
    }


@app.post("/load")
def load_video(req: LoadRequest):
    try:
        video_id = get_video_id(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if video_id in loaded_videos:
        return {
            "video_id": video_id,
            "title":    loaded_videos[video_id]["title"],
            "chunks":   loaded_videos[video_id]["chunks"],
            "status":   "already_loaded",
        }

    # Step 1: fetch transcript
    clean = clean_url(req.url)
    transcript, method = fetch_transcript_api(video_id)
    if not transcript:
        print(f"[Load] No captions — trying Whisper...")
        transcript, method = fetch_transcript_whisper(clean, video_id)
    if not transcript:
        raise HTTPException(status_code=422, detail=(
            "Could not get transcript. "
            "This video has no captions. "
            "To enable audio transcription, install ffmpeg: 'winget install ffmpeg' on Windows or 'brew install ffmpeg' on Mac, then restart the server."
        ))

    # Step 2: chunk + embed + store
    title  = get_title(video_id)
    chunks = chunk_text(transcript)
    print(f"[Load] '{title}' | {method} | {len(chunks)} chunks")

    cname = f"yt_{video_id}"
    try:
        chroma.delete_collection(cname)
    except Exception:
        pass
    col = chroma.create_collection(cname, metadata={"hnsw:space": "cosine"})

    # parallel embed all chunks at once, then batch insert into ChromaDB
    print(f"[Load] Embedding {len(chunks)} chunks in parallel...")
    embeddings = embed_chunks_parallel(chunks, max_workers=4)
    col.add(
        ids       = [f"chunk_{i}" for i in range(len(chunks))],
        embeddings= embeddings,
        documents = chunks,
        metadatas = [{"i": i} for i in range(len(chunks))],
    )
    print(f"[Load] Embeddings stored.")

    loaded_videos[video_id] = {
        "title":  title,
        "url":    req.url,
        "chunks": len(chunks),
        "method": method,
    }
    save_metadata()  # persist immediately after each load
    print(f"[Load] Done.")
    return {"video_id": video_id, "title": title, "chunks": len(chunks), "method": method, "status": "loaded"}


@app.post("/ask")
def ask_question(req: AskRequest):
    # accept full URL or video_id
    video_id = req.video_id
    try:
        video_id = get_video_id(req.video_id)
    except ValueError:
        pass  # already a plain video_id

    if video_id not in loaded_videos:
        available = [{"video_id": k, "title": v["title"]} for k, v in loaded_videos.items()]
        raise HTTPException(
            status_code=404,
            detail={"message": "Video not loaded. Call POST /load first.", "loaded_videos": available}
        )
    req.video_id = video_id

    try:
        col = chroma.get_collection(f"yt_{video_id}")
    except Exception:
        raise HTTPException(status_code=404, detail="Video data not found in store.")

    # RAG: retrieve relevant chunks
    results = col.query(query_embeddings=[embed(req.question)], n_results=req.k)
    context = "\n\n---\n\n".join(results["documents"][0])
    title   = loaded_videos[video_id]["title"]

    # init conversation history for this video if first time
    if video_id not in conversation_history:
        conversation_history[video_id] = []

    history = conversation_history[video_id]

    # system prompt + transcript context + full conversation history
    messages = [
        {
            "role": "system",
            "content": (
                f"You are a helpful assistant answering questions about the YouTube video: '{title}'. "
                "Answer using ONLY the provided transcript excerpts and conversation history. "
                "Be specific. If the answer is not in the transcript, say so clearly."
            ),
        },
        {
            "role": "user",
            "content": f"Relevant transcript excerpts:\n{context}",
        },
        {
            "role": "assistant",
            "content": "I have read the transcript excerpts. Ask me anything about the video.",
        },
    ] + history + [
        {
            "role": "user",
            "content": req.question,
        }
    ]

    full_answer = []

    def stream():
        r = client.chat.completions.create(
            model=MODEL, messages=messages, stream=True, extra_body={"keep_alive": -1}
        )
        for chunk in r:
            d = chunk.choices[0].delta
            if d and d.content:
                full_answer.append(d.content)
                yield d.content

        # save to history after streaming completes
        history.append({"role": "user",      "content": req.question})
        history.append({"role": "assistant", "content": "".join(full_answer)})

        # keep last 10 exchanges (20 messages) to avoid context overflow
        if len(history) > 20:
            conversation_history[video_id] = history[-20:]

    return StreamingResponse(stream(), media_type="text/plain")


@app.get("/videos")
def list_videos():
    return {
        "count":  len(loaded_videos),
        "videos": [{"video_id": k, **v} for k, v in loaded_videos.items()],
    }


@app.get("/status/{video_id}")
def status(video_id: str):
    if video_id in loaded_videos:
        return {"loaded": True, **loaded_videos[video_id]}
    return {"loaded": False, "video_id": video_id}


@app.get("/content/{video_id}")
def get_content(video_id: str):
    # accept full URL too
    try:
        video_id = get_video_id(video_id)
    except ValueError:
        pass

    if video_id not in loaded_videos:
        raise HTTPException(status_code=404, detail="Video not loaded. Call POST /load first.")

    try:
        col = chroma.get_collection(f"yt_{video_id}")
    except Exception:
        raise HTTPException(status_code=404, detail="Video data not found in store.")

    # fetch all chunks stored for this video
    results = col.get(include=["documents", "metadatas"])
    chunks  = results["documents"]
    metas   = results["metadatas"]

    # sort by chunk index
    paired = sorted(zip(metas, chunks), key=lambda x: x[0].get("i", 0))

    return {
        "video_id":    video_id,
        "title":       loaded_videos[video_id]["title"],
        "url":         loaded_videos[video_id]["url"],
        "method":      loaded_videos[video_id]["method"],
        "total_chunks": len(chunks),
        "full_transcript": " ".join([c for _, c in paired]),
        "chunks": [
            {"chunk_index": m.get("i", idx), "text": c}
            for idx, (m, c) in enumerate(paired)
        ],
    }


@app.delete("/clear/{video_id}")
def clear_history(video_id: str):
    try:
        video_id = get_video_id(video_id)
    except ValueError:
        pass
    if video_id in conversation_history:
        conversation_history[video_id] = []
    return {"video_id": video_id, "status": "conversation cleared"}


@app.delete("/delete/{video_id}")
def delete_video(video_id: str):
    try:
        video_id = get_video_id(video_id)
    except ValueError:
        pass

    if video_id not in loaded_videos:
        raise HTTPException(status_code=404, detail="Video not found.")

    # remove from ChromaDB
    try:
        chroma.delete_collection(f"yt_{video_id}")
    except Exception:
        pass

    # remove from memory
    loaded_videos.pop(video_id, None)
    conversation_history.pop(video_id, None)

    # persist updated metadata
    save_metadata()

    return {"video_id": video_id, "status": "deleted"}


if __name__ == "__main__":
    import uvicorn
    # use the actual filename without .py extension
    module = os.path.splitext(os.path.basename(__file__))[0]
    uvicorn.run(f"{module}:app", host="0.0.0.0", port=8000, reload=True)
