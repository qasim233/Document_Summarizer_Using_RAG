# Document Summarizer

A Python-based document summarization system that uses RAG (Retrieval-Augmented Generation) to generate comprehensive summaries of documents. The system supports PDF, TXT, and Markdown files.

## Features

- Document parsing for PDF, TXT, and Markdown files
- Text chunking with configurable chunk size and overlap
- Semantic search using FAISS and sentence transformers
- Abstractive summarization using BART
- Interactive command-line interface
- Results saving capability

## Requirements

- Python 3.8 or higher
- PyTorch
- CUDA (optional, for GPU acceleration)

## Installation

1. Create and activate a virtual environment (Already Created just activate the Environment):
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the script:
```bash
python Text_Summarizer.py
```

Note: If it gives error about "No Module" then install them manually using pip after activating environment(.venv). 

2. Follow the interactive prompts:
   - Enter the path to your document (PDF, TXT, or MD)
   - Configure chunk size (default: 500 words)
   - Configure chunk overlap (default: 100 words)
   - Set number of chunks to retrieve (default: 5)
   - Choose between default or custom models
   - Choose whether to save results

3. The system will:
   - Parse and chunk your document
   - Create embeddings for the chunks
   - Retrieve the most relevant chunks
   - Generate a summary
   - Display results and save them if requested

## Configuration Options

- **Chunk Size**: Number of words per chunk (50-2000)
- **Chunk Overlap**: Number of overlapping words between chunks
- **Chunks to Retrieve**: Number of most relevant chunks to use for summarization (1-20)
- **Models**:
  - Default: sentence-transformers/all-mpnet-base-v2 (embedding) and facebook/bart-large-cnn (summarization)
  - Custom: Specify your own models

## Output

The system generates:
1. A comprehensive summary of the document
2. The most relevant chunks used for summarization
3. Processing statistics (time, number of chunks, etc.)
4. A detailed report file (if saving is enabled)

## Project Structure

```
Document_Summarizer/
├── Text_Summarizer.py      # Main script
├──src
   ├── document_parser.py      # Document parsing module
   ├── embedding_engine.py     # Embedding generation module
   ├── summary_generator.py    # Summary generation module
├──Report
   ├── Document_Summarizer_Report.pdf
├── Documents                  # Input Document for testing the system
   ├── AI_In_Programming.txt
   ├── Artificial_Intelligence.pdf
   ├── Neuroscience_Evolution.md
├── Result                     # Output summarized Files
   ├── AI_In_Programming_summary_2025-06-10T23-32-14.txt
   ├── Artificial_Intelligence_summary_2025-06-10T23-33-18.txt
   ├── Neuroscience_Evolution_summary_2025-06-10T23-34-18.txt
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Notes

- The first run will download the required models, which may take some time
- Processing time depends on document size and available computational resources
- For best results, use documents with clear structure and formatting