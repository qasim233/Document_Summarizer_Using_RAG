import os
import time
import argparse
import numpy as np
import torch
from pathlib import Path

# Import modularized classes
from src.document_parser import DocumentParser
from src.embedding_engine import EmbeddingEngine
from src.summary_generator import SummaryGenerator

class RAGSummarizer:
    """Main RAG pipeline for document summarization"""
    
    def __init__(
        self, 
        embedding_model = 'sentence-transformers/all-mpnet-base-v2',
        summary_model = 'facebook/bart-large-cnn',
        chunk_size = 500,
        chunk_overlap = 100
    ):
        self.parser = DocumentParser(chunk_size, chunk_overlap)
        self.embedding_engine = EmbeddingEngine(embedding_model)
        self.summary_generator = SummaryGenerator(summary_model)
        self.chunks = []
        self.embeddings = None
        
    def process_document(self, file_path):
        """Process document through the RAG pipeline"""
        start_time = time.time()
        
        # Parse document
        print(f"Processing document: {file_path}")
        document_text = self.parser.parse_document(file_path)
        
        # Chunk document
        self.chunks = self.parser.chunk_document(document_text)
        print(f"Document split into {len(self.chunks)} chunks")
        
        # Create embeddings and build index
        self.embeddings = self.embedding_engine.create_embeddings(self.chunks)
        self.embedding_engine.build_faiss_index(self.embeddings)
        
        # Retrieve relevant chunks for summarization
        query = "Summarize this document"
        distances, indices = self.embedding_engine.search(query, k=5)
        
        # Prepare context for summary generation
        retrieved_chunks = [self.chunks[i] for i in indices]
        context = " ".join(retrieved_chunks)
        
        # Generate summary
        summary = self.summary_generator.generate_summary(context)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Prepare results
        result = {
            "file_path": file_path,
            "num_chunks": len(self.chunks),
            "retrieved_chunks": retrieved_chunks,
            "summary": summary,
            "processing_time": total_time,
            "similarity_scores": distances.tolist()
        }
        
        return result
    
    def display_results(self, result):
        """Display summarization results"""
        print("\n" + "="*80)
        print(f"DOCUMENT SUMMARIZATION RESULTS")
        print("="*80)
        print(f"File: {result['file_path']}")
        print(f"Processing time: {result['processing_time']:.2f} seconds")
        print(f"Number of chunks: {result['num_chunks']}")
        print("\n" + "-"*80)
        print("SUMMARY:")
        print("-"*80)
        print(result['summary'])
        print("\n" + "-"*80)
        print("TOP RETRIEVED CHUNKS:")
        print("-"*80)
        for i, chunk in enumerate(result['retrieved_chunks']):
            print(f"Chunk {i+1} (Similarity: {result['similarity_scores'][i]:.4f}):")
            print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
            print()

def get_user_input():
    """Get user input interactively"""
    print("=" * 60)
    print("RAG Document Summarization System")
    print("=" * 60)
    
    # Get file path
    while True:
        file_path = input("\nEnter the path to your document (PDF, TXT, or MD): ").strip()
        
        if not file_path:
            print("Please enter a valid file path.")
            continue
            
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        # Check file extension
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in ['.pdf', '.txt', '.md']:
            print(f"Unsupported file format: {file_ext}")
            print("   Supported formats: PDF, TXT, MD")
            continue
            
        break
    
    print(f"Selected file: {Path(file_path).name}")
    
    # Get chunk size
    while True:
        try:
            chunk_size_input = input(f"\nEnter chunk size in words (default: 500): ").strip()
            if not chunk_size_input:
                chunk_size = 500
            else:
                chunk_size = int(chunk_size_input)
                if chunk_size < 50 or chunk_size > 2000:
                    print("Chunk size should be between 50 and 2000 words.")
                    continue
            break
        except ValueError:
            print("Please enter a valid number.")
    
    # Get chunk overlap
    while True:
        try:
            overlap_input = input(f"\nEnter chunk overlap in words (default: 100): ").strip()
            if not overlap_input:
                chunk_overlap = 100
            else:
                chunk_overlap = int(overlap_input)
                if chunk_overlap < 0 or chunk_overlap >= chunk_size:
                    print(f"Chunk overlap should be between 0 and {chunk_size-1} words.")
                    continue
            break
        except ValueError:
            print("Please enter a valid number.")
    
    # Get number of chunks to retrieve
    while True:
        try:
            top_k_input = input(f"\nEnter number of chunks to retrieve (default: 5): ").strip()
            if not top_k_input:
                top_k = 5
            else:
                top_k = int(top_k_input)
                if top_k < 1 or top_k > 20:
                    print("Number of chunks should be between 1 and 20.")
                    continue
            break
        except ValueError:
            print("Please enter a valid number.")
    
    # Ask about model selection
    print(f"\nModel Selection:")
    print("1. Default models (recommended)")
    print("2. Custom models")
    
    while True:
        model_choice = input("Choose option (1 or 2, default: 1): ").strip()
        if not model_choice or model_choice == '1':
            embedding_model = 'sentence-transformers/all-mpnet-base-v2'
            summary_model = 'facebook/bart-large-cnn'
            break
        elif model_choice == '2':
            embedding_model = input("Enter embedding model name (default: sentence-transformers/all-mpnet-base-v2): ").strip()
            if not embedding_model:
                embedding_model = 'sentence-transformers/all-mpnet-base-v2'
            
            summary_model = input("Enter summary model name (default: facebook/bart-large-cnn): ").strip()
            if not summary_model:
                summary_model = 'facebook/bart-large-cnn'
            break
        else:
            print("Please enter 1 or 2.")
    
    # Ask about saving results
    save_results = input(f"\nSave results to file? (y/n, default: y): ").strip().lower()
    save_results = save_results != 'n'
    
    return {
        'file_path': file_path,
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'top_k': top_k,
        'embedding_model': embedding_model,
        'summary_model': summary_model,
        'save_results': save_results
    }

def display_configuration(config):
    """Display the configuration before processing"""
    print("\n" + "=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Document: {Path(config['file_path']).name}")
    print(f"Chunk Size: {config['chunk_size']} words")
    print(f"Chunk Overlap: {config['chunk_overlap']} words")
    print(f"Chunks to Retrieve: {config['top_k']}")
    print(f"Embedding Model: {config['embedding_model']}")
    print(f"Summary Model: {config['summary_model']}")
    print(f"Save Results: {'Yes' if config['save_results'] else 'No'}")
    print("=" * 60)
    
    # Confirm before proceeding
    confirm = input("\nProceed with these settings? (y/n, default: y): ").strip().lower()
    return confirm != 'n'

def save_results_to_file(result, config):
    """Save results to a file"""
    try:
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Generate filename
        file_stem = Path(config['file_path']).stem
        timestamp = result['timestamp'].replace(':', '-').replace('.', '-')
        output_file = f"results/{file_stem}_summary_{timestamp[:19]}.txt"
        
        # Create detailed report
        report = f"""
RAG DOCUMENT SUMMARIZATION RESULTS
{'=' * 50}

DOCUMENT INFORMATION:
- File: {result['file_path']}
- Processing Time: {result['processing_time']:.2f} seconds
- Number of Chunks: {result['num_chunks']}
- Token Usage: {result['token_usage']}
- Summary Generation Latency: {result['latency']:.2f} seconds

CONFIGURATION:
- Chunk Size: {config['chunk_size']} words
- Chunk Overlap: {config['chunk_overlap']} words
- Chunks Retrieved: {config['top_k']}
- Embedding Model: {config['embedding_model']}
- Summary Model: {config['summary_model']}

GENERATED SUMMARY:
{'-' * 50}
{result['summary']}

RETRIEVED CONTEXT CHUNKS:
{'-' * 50}
"""
        
        for i, chunk in enumerate(result['retrieved_chunks'], 1):
            similarity = result['similarity_scores'][i-1]
            report += f"\nChunk {i} (Similarity: {similarity:.4f}):\n"
            report += f"{chunk}\n"
            report += "-" * 30 + "\n"
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\nResults saved to: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        return None

def main():
    """Interactive main function"""
    try:
        # Get user input
        config = get_user_input()
        
        # Display configuration and confirm
        if not display_configuration(config):
            print("Operation cancelled by user.")
            return
        
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nUsing device: {device}")
        
        # Initialize RAG components
        print("\nInitializing RAG system...")
        
        # Initialize components directly
        parser = DocumentParser(config['chunk_size'], config['chunk_overlap'])
        embedding_engine = EmbeddingEngine(config['embedding_model'])
        summary_generator = SummaryGenerator(config['summary_model'])
        
        print("RAG system initialized successfully!")
        
        # Process document
        print(f"\nProcessing document: {Path(config['file_path']).name}")
        
        start_time = time.time()
        
        # Parse and chunk document
        print("Parsing document...")
        document_text = parser.parse_document(config['file_path'])
        
        print("Chunking document...")
        chunks = parser.chunk_document(document_text)
        print(f"Document split into {len(chunks)} chunks")
        
        # Create embeddings
        print("Creating embeddings...")
        embeddings = embedding_engine.create_embeddings(chunks)
        embedding_engine.build_faiss_index(embeddings)
        print("Embeddings created and indexed")
        
        # Retrieve relevant chunks
        print(f"Retrieving top {config['top_k']} relevant chunks...")
        query = "Summarize this document comprehensively"
        distances, indices = embedding_engine.search(query, k=min(config['top_k'], len(chunks)))
        
        retrieved_chunks = [chunks[i] for i in indices]
        context = " ".join(retrieved_chunks)
        print("Relevant chunks retrieved")
        
        # Generate summary
        print("Generating summary...")
        summary = summary_generator.generate_summary(context)
        print("Summary generated successfully!")
        
        # Calculate total processing time
        end_time = time.time()
        total_time = end_time - start_time
        
        # Prepare results
        from datetime import datetime
        result = {
            "file_path": config['file_path'],
            "num_chunks": len(chunks),
            "retrieved_chunks": retrieved_chunks,
            "summary": summary,
            "processing_time": total_time,
            "similarity_scores": distances.tolist(),
            "token_usage": "N/A",  # Not tracked in current implementation
            "latency": total_time,  # Approximate latency
            "timestamp": datetime.now().isoformat()
        }
        
        # Display results
        print("\n" + "=" * 80)
        print("SUMMARIZATION RESULTS")
        print("=" * 80)
        print(f"File: {Path(result['file_path']).name}")
        print(f"Processing time: {result['processing_time']:.2f} seconds")
        print(f"Number of chunks: {result['num_chunks']}")
        
        print("\n" + "-" * 80)
        print("GENERATED SUMMARY:")
        print("-" * 80)
        print(result['summary'])
        
        print("\n" + "-" * 80)
        print(f"TOP {len(retrieved_chunks)} RETRIEVED CHUNKS:")
        print("-" * 80)
        for i, chunk in enumerate(retrieved_chunks, 1):
            similarity = result['similarity_scores'][i-1]
            print(f"\nChunk {i} (Similarity: {similarity:.4f}):")
            print(chunk[:300] + "..." if len(chunk) > 300 else chunk)
        
        # Save results if requested
        if config['save_results']:
            save_results_to_file(result, config)
        
        print(f"\nSummarization completed successfully!")
        
        # Ask if user wants to process another document
        another = input(f"\nProcess another document? (y/n): ").strip().lower()
        if another == 'y':
            print("\n" + "=" * 60)
            main()  # Recursive call for another document
        
    except KeyboardInterrupt:
        print(f"\n\nOperation cancelled by user (Ctrl+C)")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Please check your inputs and try again.")

if __name__ == "__main__":
    main()