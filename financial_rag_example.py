"""
Financial RAG System - Notebook Example
Copy these cells into your Jupyter notebook to use the financial RAG system
"""

# CELL 1: Install Required Dependencies
"""
%pip install langchain langchain-openai langchain-community sentence-transformers rank-bm25
"""

# CELL 2: Import and Setup
"""
import os
from pathlib import Path
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

# Import financial RAG system
from financial_rag_system import (
    FinancialRAGSystem,
    FinancialDocumentProcessor,
    FinancialHybridRetriever,
    FinancialReranker
)

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Initialize components
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(
    model="gpt-4",  # Use GPT-4 for financial accuracy
    temperature=0,  # Zero temperature for consistency
    max_tokens=2000
)

print("✅ Components initialized")
"""

# CELL 3: Load and Process Financial Documents
"""
# Load financial PDFs
pdf_directory = Path("../financial_documents")
pdf_files = list(pdf_directory.glob("*.pdf"))

print(f"Found {len(pdf_files)} PDF files")

# Load all documents
all_documents = []
for pdf_file in pdf_files:
    loader = PyPDFLoader(str(pdf_file))
    docs = loader.load()
    
    # Add source metadata
    for doc in docs:
        doc.metadata['source'] = pdf_file.name
        doc.metadata['doc_id'] = pdf_file.stem
    
    all_documents.extend(docs)
    print(f"Loaded {len(docs)} pages from {pdf_file.name}")

print(f"\\nTotal documents loaded: {len(all_documents)}")

# Process documents with financial-specific chunking
processor = FinancialDocumentProcessor(
    chunk_size=800,
    chunk_overlap=200
)

processed_chunks = processor.split_with_metadata(all_documents)

print(f"✅ Created {len(processed_chunks)} chunks with financial metadata")

# Show sample chunk metadata
if processed_chunks:
    sample = processed_chunks[0]
    print("\\nSample chunk metadata:")
    print(f"  - Contains numbers: {sample.metadata.get('contains_numbers')}")
    print(f"  - Number count: {sample.metadata.get('number_count')}")
    print(f"  - Numbers found: {sample.metadata.get('numbers', [])}")
    print(f"  - Dates found: {sample.metadata.get('dates', [])}")
"""

# CELL 4: Create Vector Store
"""
# Create Chroma vector store with persistence
persist_directory = "./financial_chroma_db"

vectorstore = Chroma.from_documents(
    documents=processed_chunks,
    embedding=embeddings,
    persist_directory=persist_directory,
    collection_name="financial_documents"
)

# Persist the database
vectorstore.persist()

print(f"✅ Vector store created with {len(processed_chunks)} chunks")
print(f"📁 Persisted to: {persist_directory}")
"""

# CELL 5: Initialize Financial RAG System
"""
# Initialize the complete financial RAG system
financial_rag = FinancialRAGSystem(
    vectorstore=vectorstore,
    embeddings=embeddings,
    llm=llm,
    documents=processed_chunks
)

print("✅ Financial RAG System initialized")
print("\\n🎯 System Features:")
print("  - Hybrid search (semantic + keyword)")
print("  - Multi-query expansion")
print("  - Parent document retrieval")
print("  - Cross-encoder reranking")
print("  - Confidence scoring")
print("  - Source citation tracking")
"""

# CELL 6: Test Queries
"""
# Test queries for financial documents
test_queries = [
    "What was the total revenue in Q4 2023?",
    "What is the operating margin for fiscal year 2023?",
    "How much cash did the company have at year end?",
    "What were the capital expenditures in 2023?",
    "What is the debt-to-equity ratio?"
]

print("🔍 Testing Financial RAG System\\n")
print("=" * 80)

for query in test_queries:
    print(f"\\n📊 Query: {query}")
    print("-" * 80)
    
    result = financial_rag.query_with_verification(
        query=query,
        min_confidence=0.70  # 70% confidence threshold
    )
    
    print(f"\\n💡 Answer:")
    print(result['answer'])
    print(f"\\n📈 Confidence: {result['confidence']:.1%}")
    print(f"📚 Sources: {result['num_sources']}")
    
    # Show sources
    for source in result['sources']:
        print(f"\\n  [Source {source['source_id']}]")
        print(f"    Page: {source['page']}")
        print(f"    Confidence: {source['confidence']:.1%}")
        print(f"    Excerpt: {source['excerpt'][:150]}...")
    
    print("\\n" + "=" * 80)
"""

# CELL 7: Interactive Query Interface
"""
def query_financial_documents(question: str, min_confidence: float = 0.75):
    \"\"\"Interactive function to query financial documents\"\"\"
    
    print(f"\\n🔍 Processing query: {question}")
    print("=" * 80)
    
    result = financial_rag.query_with_verification(
        query=question,
        min_confidence=min_confidence
    )
    
    # Display results
    print(f"\\n💡 ANSWER:")
    print("-" * 80)
    print(result['answer'])
    
    print(f"\\n📊 METADATA:")
    print("-" * 80)
    print(f"Confidence Score: {result['confidence']:.1%}")
    print(f"Number of Sources: {result['num_sources']}")
    print(f"Timestamp: {result['timestamp']}")
    
    if result.get('warning'):
        print(f"⚠️  Warning: {result['warning']}")
    
    print(f"\\n📚 SOURCES:")
    print("-" * 80)
    for i, source in enumerate(result['sources'], 1):
        print(f"\\n[{i}] Page {source['page']} (Confidence: {source['confidence']:.1%})")
        print(f"    {source['excerpt'][:200]}...")
        
        # Show numbers if present
        if source['metadata'].get('numbers'):
            print(f"    Numbers found: {', '.join(source['metadata']['numbers'][:5])}")
        
        # Show dates if present
        if source['metadata'].get('dates'):
            print(f"    Dates found: {', '.join(source['metadata']['dates'][:3])}")
    
    return result

# Example usage
result = query_financial_documents(
    "What was the net income for fiscal year 2023?",
    min_confidence=0.75
)
"""

# CELL 8: Batch Processing
"""
# Process multiple queries at once
batch_queries = [
    "What is the current ratio?",
    "What were the total assets at year end?",
    "What is the return on equity?",
    "How much did the company spend on R&D?",
    "What is the dividend per share?"
]

print("🔄 Processing batch queries...")
print("=" * 80)

batch_results = financial_rag.batch_query(batch_queries)

# Create summary report
import pandas as pd

summary_data = []
for query, result in zip(batch_queries, batch_results):
    summary_data.append({
        'Query': query[:50] + '...' if len(query) > 50 else query,
        'Confidence': f"{result['confidence']:.1%}",
        'Sources': result['num_sources'],
        'Has_Answer': 'Yes' if result['confidence'] > 0.5 else 'No'
    })

summary_df = pd.DataFrame(summary_data)
print("\\n📊 Batch Query Summary:")
print(summary_df.to_string(index=False))

# Show detailed results
print("\\n" + "=" * 80)
print("📋 DETAILED RESULTS:")
print("=" * 80)

for query, result in zip(batch_queries, batch_results):
    print(f"\\n❓ {query}")
    print(f"💡 {result['answer'][:200]}...")
    print(f"📈 Confidence: {result['confidence']:.1%}")
    print("-" * 80)
"""

# CELL 9: Export Results
"""
# Export query results to file
import json
from datetime import datetime

def export_results(results: list, filename: str = None):
    \"\"\"Export query results to JSON file\"\"\"
    
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"financial_rag_results_{timestamp}.json"
    
    # Prepare data for export
    export_data = {
        'export_date': datetime.now().isoformat(),
        'total_queries': len(results),
        'results': results
    }
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"✅ Results exported to: {filename}")
    return filename

# Export batch results
export_file = export_results(batch_results)

# Also create CSV summary
summary_csv = summary_df.copy()
summary_csv['Export_Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
csv_filename = f"financial_rag_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
summary_csv.to_csv(csv_filename, index=False)

print(f"✅ Summary exported to: {csv_filename}")
"""

# CELL 10: System Statistics and Monitoring
"""
# Get system statistics
def show_system_stats():
    \"\"\"Display comprehensive system statistics\"\"\"
    
    print("📊 FINANCIAL RAG SYSTEM STATISTICS")
    print("=" * 80)
    
    # Vector store stats
    collection = vectorstore._collection
    print(f"\\n📚 Vector Store:")
    print(f"  - Total chunks: {collection.count()}")
    print(f"  - Collection name: {collection.name}")
    
    # Document stats
    print(f"\\n📄 Documents:")
    print(f"  - Total documents: {len(all_documents)}")
    print(f"  - Total chunks: {len(processed_chunks)}")
    print(f"  - Avg chunks per doc: {len(processed_chunks) / len(all_documents):.1f}")
    
    # Metadata analysis
    chunks_with_numbers = sum(1 for c in processed_chunks if c.metadata.get('contains_numbers'))
    chunks_with_dates = sum(1 for c in processed_chunks if c.metadata.get('contains_dates'))
    
    print(f"\\n🔢 Content Analysis:")
    print(f"  - Chunks with numbers: {chunks_with_numbers} ({chunks_with_numbers/len(processed_chunks)*100:.1f}%)")
    print(f"  - Chunks with dates: {chunks_with_dates} ({chunks_with_dates/len(processed_chunks)*100:.1f}%)")
    
    # Retriever info
    print(f"\\n🔍 Retrievers:")
    print(f"  - Hybrid retriever: ✅ Active")
    print(f"  - Multi-query retriever: ✅ Active")
    print(f"  - Parent retriever: ✅ Active")
    print(f"  - Reranker: {'✅ Active' if financial_rag.reranker.enabled else '❌ Disabled'}")
    
    print("\\n" + "=" * 80)

show_system_stats()
"""

# CELL 11: Advanced Testing
"""
# Test retrieval quality
def test_retrieval_quality(query: str):
    \"\"\"Test and compare different retrieval methods\"\"\"
    
    print(f"🧪 Testing retrieval quality for: {query}")
    print("=" * 80)
    
    # Test hybrid retrieval
    hybrid_docs = financial_rag.hybrid_retriever.retrieve(query, k=5)
    print(f"\\n1️⃣ Hybrid Retrieval: {len(hybrid_docs)} documents")
    for i, doc in enumerate(hybrid_docs[:3], 1):
        print(f"   [{i}] {doc.page_content[:100]}...")
    
    # Test multi-query retrieval
    multi_docs = financial_rag.multi_query_retriever.retrieve(query)
    print(f"\\n2️⃣ Multi-Query Retrieval: {len(multi_docs)} documents")
    for i, doc in enumerate(multi_docs[:3], 1):
        print(f"   [{i}] {doc.page_content[:100]}...")
    
    # Test with reranking
    all_docs = financial_rag._deduplicate_documents(hybrid_docs + multi_docs)
    reranked = financial_rag.reranker.rerank(query, all_docs, top_k=5)
    print(f"\\n3️⃣ After Reranking: {len(reranked)} documents")
    for i, result in enumerate(reranked[:3], 1):
        print(f"   [{i}] Score: {result['score']:.3f}")
        print(f"       {result['content'][:100]}...")
    
    print("\\n" + "=" * 80)

# Test with a sample query
test_retrieval_quality("What was the revenue growth rate in 2023?")
"""

print(__doc__)
