"""
Service RAG avec ChromaDB
Recherche documentaire dans la FAQ interne.
"""

import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import settings


class RAGService:
    _instance = None
    _retriever = None
    _vector_store = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialise ChromaDB avec la FAQ."""
        persist_dir = "app/data/chroma_db"
        
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL
        )
        
        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            self._vector_store = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings
            )
        else:
            self._create_chromadb(embeddings, persist_dir)
        
        self._retriever = self._vector_store.as_retriever(
            search_kwargs={"k": settings.TOP_K_RESULTS}
        )

    def _create_chromadb(self, embeddings, persist_dir):
        """Cree et indexe ChromaDB a partir du fichier FAQ."""
        with open(settings.FAQ_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        
        chunks = self._extract_sections(content)
        
        self._vector_store = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            persist_directory=persist_dir
        )

    def _extract_sections(self, content: str) -> list:
        """Extrait les règles du fichier."""
        chunks = []
        current_rule = ""
        
        for line in content.split('\n'):
            if line.strip().startswith('- Règle'):
                if current_rule:
                    chunks.append(current_rule)
                current_rule = line + "\n"
            else:
                current_rule += line + "\n"
        
        if current_rule:
            chunks.append(current_rule)
        
        return chunks or [content]

    def query(self, question: str, top_k: int = None) -> list:
        """Recherche les chunks les plus pertinents via ChromaDB."""
        if top_k is None:
            top_k = settings.TOP_K_RESULTS
        
        self._retriever.search_kwargs["k"] = top_k
        docs = self._retriever.invoke(question)
        return [doc.page_content for doc in docs]

