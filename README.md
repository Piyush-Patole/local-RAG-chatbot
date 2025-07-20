A **local RAG (Retrieval-Augmented Generation)** chatbot that answers questions based on custom documents (PDFs) without needing an internet connection. 

**Key features:**
1. 100% Local Execution: Runs entirely on your machine (CPU-only)
2. Document Intelligence: Reads and understands PDF files
3. Memory Efficient: Optimized for 8GB RAM systems
4. Conversational Interface: Interactive Q&A with streaming responses

**Technical Stack used:**
1. **Framework:** LangChain
2. **Embeddings:** HuggingFace all-MiniLM-L6-v2 (Lightweight (80MB), CPU-friendly)
3. **Vector DB:** Chroma
4. **LLM:** Zephyr-7B(4-bit quantized) (Balanced performance for 8GB RAM)
5. **Document Loader:** PyPDFLoader

**Working (Step-by-Step):**
**A. Document Processing Pipeline:**
1. **Loading:** Scans docs/ folder for PDFs using DirectoryLoader. Extracts text with PyPDFLoader.
2. **Chunking:** Splits documents into 300-character segments with 50-character overlaps. Uses RecursiveCharacterTextSplitter to preserve context.
3. **Vectorization:** Converts text to numerical vectors using all-MiniLM-L6-v2 embeddings. Stores vectors in ChromaDB for fast searching.

**B. Question-Answering System:**
1. **Retrieval:** When a question is asked => Converts question to vector. Finds  most relevant document chunks.
2. **Generation:** Combines question + retrieved context in the prompt by Zephyr-7B model.
3. **Streaming Interface:** Displays answers word-by-word for natural conversation feel.

**Technical Challenges & Solutions:**
1. **Limited RAM:** Used quantized Zephyr-7B (4-bit) and small embeddings.
2. **Ollama connectivity issues:** Switched to fully local CTransformers backend.
3. **PDF parsing complexity:** Implemented recursive text splitting with overlaps.
