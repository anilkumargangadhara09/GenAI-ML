# Financial RAG System - Production Implementation

## 🎯 Overview

A production-ready RAG (Retrieval-Augmented Generation) system specifically designed for financial documents where **accuracy is critical** and missing a single number could cost millions.

## ✨ Key Features

### 1. **Hybrid Search Architecture**
- **Dense Retrieval**: Semantic search using embeddings
- **Sparse Retrieval**: Keyword-based BM25 search
- **Ensemble**: Combines both for maximum recall

### 2. **Multi-Query Expansion**
- Automatically generates query variations
- Focuses on numbers, dates, entities, and context
- Reduces risk of missing relevant information

### 3. **Parent Document Retrieval**
- Searches small, precise chunks
- Returns larger context for better understanding
- Preserves surrounding information

### 4. **Cross-Encoder Reranking**
- Re-scores retrieved documents for relevance
- Improves precision significantly
- Filters low-quality results

### 5. **Financial-Specific Processing**
- Extracts and preserves numbers exactly
- Identifies dates and time periods
- Tracks financial entities (tickers, accounts)
- Maintains decimal precision

### 6. **Confidence Scoring**
- Every result has a confidence score
- Configurable confidence thresholds
- Warns when confidence is low

### 7. **Source Citation**
- Every fact is cited with source
- Tracks page numbers and metadata
- Enables manual verification

## 📦 Installation

```bash
pip install langchain langchain-openai langchain-community sentence-transformers rank-bm25 chromadb pypdf
```

## 🚀 Quick Start

### Basic Usage

```python
from financial_rag_system import FinancialRAGSystem, FinancialDocumentProcessor
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

# 1. Load documents
loader = PyPDFLoader("financial_report.pdf")
documents = loader.load()

# 2. Process with financial-specific chunking
processor = FinancialDocumentProcessor(chunk_size=800, chunk_overlap=200)
chunks = processor.split_with_metadata(documents)

# 3. Create vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings()
)

# 4. Initialize RAG system
financial_rag = FinancialRAGSystem(
    vectorstore=vectorstore,
    embeddings=OpenAIEmbeddings(),
    llm=ChatOpenAI(model="gpt-4", temperature=0),
    documents=chunks
)

# 5. Query with verification
result = financial_rag.query_with_verification(
    "What was the Q4 2023 revenue?",
    min_confidence=0.75
)

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Sources: {result['num_sources']}")
```

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Financial RAG System                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Document Processing                        │
│  • Extract numbers, dates, entities                         │
│  • Smart chunking with overlap                              │
│  • Metadata enrichment                                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Multi-Stage Retrieval                      │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Hybrid     │  │ Multi-Query  │  │   Parent     │     │
│  │  Retrieval   │  │  Expansion   │  │  Document    │     │
│  │ (Dense+BM25) │  │              │  │  Retrieval   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Cross-Encoder Reranking                   │
│  • Score all retrieved documents                            │
│  • Sort by relevance                                        │
│  • Filter by confidence threshold                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Answer Generation                         │
│  • Build context with source citations                      │
│  • Generate answer with LLM                                 │
│  • Preserve exact numbers and dates                         │
│  • Return with confidence scores                            │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Configuration

### Chunk Size Optimization

```python
# For dense financial text
processor = FinancialDocumentProcessor(
    chunk_size=600,
    chunk_overlap=150
)

# For sparse documents with tables
processor = FinancialDocumentProcessor(
    chunk_size=1000,
    chunk_overlap=250
)
```

### Retrieval Weights

```python
# Favor semantic search
hybrid_retriever = FinancialHybridRetriever(
    vectorstore, documents, 
    weights=[0.7, 0.3]  # [dense, sparse]
)

# Favor keyword search
hybrid_retriever = FinancialHybridRetriever(
    vectorstore, documents,
    weights=[0.3, 0.7]
)
```

### Confidence Thresholds

```python
# High precision (fewer but more accurate results)
result = financial_rag.query_with_verification(
    query, min_confidence=0.85
)

# High recall (more results, some may be less relevant)
result = financial_rag.query_with_verification(
    query, min_confidence=0.60
)
```

## 📈 Performance Optimization

### 1. **Vector Store Persistence**

```python
# Save vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./financial_db"
)
vectorstore.persist()

# Load existing vector store
vectorstore = Chroma(
    persist_directory="./financial_db",
    embedding_function=embeddings
)
```

### 2. **Batch Processing**

```python
queries = [
    "What was the revenue?",
    "What was the net income?",
    "What is the debt ratio?"
]

results = financial_rag.batch_query(queries)
```

### 3. **Caching**

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_query(query: str):
    return financial_rag.query_with_verification(query)
```

## 🧪 Testing & Validation

### Test Retrieval Quality

```python
def test_retrieval(query: str):
    # Get documents
    docs = financial_rag.retrieve_with_confidence(query)
    
    # Check confidence
    avg_confidence = sum(d['confidence'] for d in docs) / len(docs)
    
    # Check if numbers are preserved
    for doc in docs:
        numbers = processor.extract_numbers(doc['content'])
        print(f"Found numbers: {numbers}")
    
    return docs
```

### Validate Answers

```python
def validate_answer(result: dict):
    """Validate answer quality"""
    
    checks = {
        'has_sources': result['num_sources'] > 0,
        'high_confidence': result['confidence'] > 0.7,
        'has_citations': '[Source' in result['answer'],
        'no_errors': 'error' not in result
    }
    
    return all(checks.values()), checks
```

## 🎯 Best Practices

### 1. **Always Use GPT-4 for Financial Data**
```python
llm = ChatOpenAI(
    model="gpt-4",  # More accurate than GPT-3.5
    temperature=0,  # Deterministic outputs
    max_tokens=2000
)
```

### 2. **Set Appropriate Confidence Thresholds**
- **Critical decisions**: 0.85+
- **General queries**: 0.70-0.85
- **Exploratory**: 0.60-0.70

### 3. **Always Verify High-Stakes Information**
```python
result = financial_rag.query_with_verification(query)

if result['confidence'] < 0.90:
    print("⚠️ Manual verification recommended")
    
# Show sources for verification
for source in result['sources']:
    print(f"Page {source['page']}: {source['excerpt']}")
```

### 4. **Monitor and Log All Queries**
```python
import logging

logging.basicConfig(
    filename='financial_rag.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# Log every query
logging.info(f"Query: {query}, Confidence: {result['confidence']}")
```

### 5. **Regular System Audits**
- Review low-confidence queries
- Check for missing information
- Update documents regularly
- Validate number extraction

## 🔒 Security Considerations

1. **API Key Management**
```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

2. **Data Privacy**
- Don't log sensitive financial data
- Use local vector stores for confidential documents
- Implement access controls

3. **Audit Trail**
```python
result['timestamp'] = datetime.now().isoformat()
result['user'] = current_user
result['query_hash'] = hash(query)
```

## 📊 Monitoring & Metrics

### Key Metrics to Track

```python
metrics = {
    'avg_confidence': result['confidence'],
    'num_sources': result['num_sources'],
    'response_time': time_taken,
    'query_length': len(query),
    'answer_length': len(result['answer'])
}
```

### Performance Dashboard

```python
import pandas as pd

def create_dashboard(results: list):
    df = pd.DataFrame([
        {
            'Query': r['query'][:50],
            'Confidence': f"{r['confidence']:.1%}",
            'Sources': r['num_sources'],
            'Timestamp': r['timestamp']
        }
        for r in results
    ])
    
    print(df.to_string())
    return df
```

## 🐛 Troubleshooting

### Issue: Low Confidence Scores

**Solution:**
- Increase chunk overlap
- Adjust retrieval weights
- Use more specific queries
- Check document quality

### Issue: Missing Numbers

**Solution:**
- Verify PDF extraction quality
- Check chunk boundaries
- Increase chunk size
- Use parent document retrieval

### Issue: Slow Performance

**Solution:**
- Use vector store persistence
- Reduce number of retrievers
- Implement caching
- Batch process queries

## 📚 Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [ChromaDB](https://www.trychroma.com/)

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Enhanced entity extraction
- Table parsing
- Multi-language support
- Advanced reranking models

## 📄 License

MIT License - See LICENSE file for details

## 📧 Support

For issues or questions, please open an issue on GitHub or contact the development team.

---

**⚠️ Important Disclaimer**: This system is designed to assist with financial document analysis but should not be the sole basis for financial decisions. Always verify critical information manually and consult with qualified financial professionals.
