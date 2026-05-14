"""
Handling document updates and maintaining freshness is crucial for production RAG systems. Here are the key strategies and best practices:
Best Practices Summary
    Implement Change Detection: Use file hashes, timestamps, or database triggers
    Use Incremental Updates: Only update changed documents, not the entire index
    Version Control: Keep document versions for rollback capability
    Smart Caching: Implement cache invalidation tied to document updates
    Automated Monitoring: Set up schedulers and file watchers
    Graceful Degradation: Handle update failures without breaking the system
    Performance Optimization: Batch updates and use background processing
    Audit Logging: Track all document changes and updates
    This approach ensures your RAG system stays fresh while maintaining performance and reliability!

🔧 Key Production Features:
  📊 Automatic Monitoring:
    File system watcher detects PDF changes in real-time
    Hash-based change detection prevents unnecessary reprocessing
    Metadata tracking for all documents
  ⏰ Scheduled Operations:
    Every 30 minutes: Process queued updates
    Every 2 hours: Full document scan
    Daily at 3:00 AM: Cleanup deleted files
  🔄 Smart Updates:
    Incremental updates (only changed documents)
    Old chunk removal before adding new ones
    Persistent metadata tracking
  
  📈 System Dashboard:
    Real-time statistics
    Service health monitoring
    Queue status tracking
  🎯 Production Benefits:
    Zero-downtime updates - Background processing doesn't affect users
    Efficient resource usage - Only processes changed documents
    Automatic cleanup - Handles deleted files gracefully
    Scalable architecture - Handles hundreds of documents
    Monitoring ready - Built-in health checks and statistics

  📋 Quick Integration:
    Copy the code from 
    production_usage_example.py
    into new notebook cells to:
  
    Enable production features
    Add enhanced Gradio interface with status monitoring
    Set up health checks and alerting
"""
"""
Production Document Manager for RAG Systems
Advanced document management with automatic updates and freshness management
"""

import json
import schedule
import threading
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configure logging
logger = logging.getLogger(__name__)

class ProductionDocumentManager:
    """Production-ready document management with automatic updates"""
    
    def __init__(self, vectorstore, embeddings, document_directory: str):
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.document_directory = Path(document_directory)
        self.metadata_file = "document_metadata.json"
        self.document_metadata = self._load_metadata()
        self.update_queue = []
        self.is_running = False
        self.observer = None
        
        logger.info(f"ProductionDocumentManager initialized for: {document_directory}")
    
    def _load_metadata(self) -> Dict:
        """Load document metadata from file"""
        try:
            if Path(self.metadata_file).exists():
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load metadata: {e}")
        return {}
    
    def _save_metadata(self):
        """Save document metadata to file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.document_metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Could not save metadata: {e}")
    
    def get_file_hash(self, file_path: Path) -> str:
        """Generate MD5 hash for file content"""
        import hashlib
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def has_document_changed(self, file_path: Path) -> bool:
        """Check if document has changed since last processing"""
        if not file_path.exists():
            return False
            
        current_hash = self.get_file_hash(file_path)
        file_key = str(file_path)
        
        if file_key not in self.document_metadata:
            return True  # New document
        
        return current_hash != self.document_metadata[file_key].get('hash')
    
    def update_document_in_vectorstore(self, file_path: Path, chunk_size=1000, chunk_overlap=200):
        """Update a single document in the vector store"""
        try:
            from langchain_community.document_loaders import PyPDFLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            
            doc_id = file_path.stem
            logger.info(f"Updating document: {doc_id}")
            
            # Remove old chunks for this document
            self._remove_document_chunks(doc_id)
            
            # Process new document
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            full_text = "\n".join([doc.page_content for doc in documents])
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            text_chunks = text_splitter.split_text(full_text)
            
            # Add new chunks with metadata
            chunk_ids = []
            for i, chunk in enumerate(text_chunks):
                chunk_id = f"{doc_id}_chunk_{i}_{int(time.time())}"
                self.vectorstore.add_texts(
                    texts=[chunk],
                    metadatas=[{
                        'doc_id': doc_id,
                        'chunk_id': chunk_id,
                        'source_file': str(file_path),
                        'last_updated': datetime.now().isoformat(),
                        'chunk_index': i
                    }],
                    ids=[chunk_id]
                )
                chunk_ids.append(chunk_id)
            
            # Update metadata
            self.document_metadata[str(file_path)] = {
                'hash': self.get_file_hash(file_path),
                'last_updated': datetime.now().isoformat(),
                'size': file_path.stat().st_size,
                'chunk_count': len(text_chunks),
                'chunk_ids': chunk_ids
            }
            
            self._save_metadata()
            logger.info(f"Successfully updated {doc_id} with {len(text_chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Failed to update document {file_path}: {e}")
    
    def _remove_document_chunks(self, doc_id: str):
        """Remove all chunks for a document"""
        try:
            # Find chunks for this document
            collection = self.vectorstore._collection
            results = collection.get(where={"doc_id": doc_id})
            
            if results['ids']:
                self.vectorstore.delete(ids=results['ids'])
                logger.info(f"Removed {len(results['ids'])} old chunks for {doc_id}")
                
        except Exception as e:
            logger.warning(f"Could not remove old chunks for {doc_id}: {e}")
    
    def scan_and_update_all(self):
        """Scan directory and update all changed documents"""
        logger.info("Starting full document scan...")
        updated_count = 0
        
        for file_path in self.document_directory.glob("*.pdf"):
            if self.has_document_changed(file_path):
                self.update_document_in_vectorstore(file_path)
                updated_count += 1
        
        logger.info(f"Document scan completed. Updated {updated_count} documents.")
        return updated_count
    
    def start_file_watcher(self):
        """Start file system watcher for real-time updates"""
        class DocumentHandler(FileSystemEventHandler):
            def __init__(self, manager):
                self.manager = manager
            
            def on_modified(self, event):
                if not event.is_directory and event.src_path.endswith('.pdf'):
                    logger.info(f"File modified: {event.src_path}")
                    self.manager.queue_update(Path(event.src_path))
            
            def on_created(self, event):
                if not event.is_directory and event.src_path.endswith('.pdf'):
                    logger.info(f"File created: {event.src_path}")
                    time.sleep(2)  # Wait for file to be fully written
                    self.manager.queue_update(Path(event.src_path))
        
        handler = DocumentHandler(self)
        self.observer = Observer()
        self.observer.schedule(handler, str(self.document_directory), recursive=True)
        self.observer.start()
        logger.info("File watcher started")
    
    def queue_update(self, file_path: Path):
        """Queue document for update"""
        if file_path not in self.update_queue:
            self.update_queue.append(file_path)
    
    def process_update_queue(self):
        """Process queued document updates"""
        if self.update_queue:
            logger.info(f"Processing {len(self.update_queue)} queued updates...")
            for file_path in self.update_queue.copy():
                if file_path.exists():
                    self.update_document_in_vectorstore(file_path)
                self.update_queue.remove(file_path)
    
    def start_scheduler(self):
        """Start background scheduler for periodic updates"""
        def schedule_worker():
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)
        
        # Schedule periodic tasks
        schedule.every(30).minutes.do(self.process_update_queue)
        schedule.every(2).hours.do(self.scan_and_update_all)
        schedule.every().day.at("03:00").do(self.cleanup_old_metadata)
        
        self.is_running = True
        scheduler_thread = threading.Thread(target=schedule_worker)
        scheduler_thread.daemon = True
        scheduler_thread.start()
        logger.info("Background scheduler started")
    
    def cleanup_old_metadata(self):
        """Clean up metadata for deleted files"""
        logger.info("Cleaning up old metadata...")
        files_to_remove = []
        
        for file_path_str in self.document_metadata:
            file_path = Path(file_path_str)
            if not file_path.exists():
                files_to_remove.append(file_path_str)
                # Remove chunks for deleted document
                self._remove_document_chunks(file_path.stem)
        
        for file_path_str in files_to_remove:
            del self.document_metadata[file_path_str]
        
        if files_to_remove:
            self._save_metadata()
            logger.info(f"Cleaned up metadata for {len(files_to_remove)} deleted files")
    
    def get_system_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        try:
            total_docs = len(self.document_metadata)
            total_chunks = sum(meta.get('chunk_count', 0) for meta in self.document_metadata.values())
            last_update = max((meta.get('last_updated', '') for meta in self.document_metadata.values()), default='Never')
            
            return {
                'total_documents': total_docs,
                'total_chunks': total_chunks,
                'vectorstore_count': self.vectorstore._collection.count(),
                'last_update': last_update,
                'queue_size': len(self.update_queue),
                'watcher_active': self.observer.is_alive() if self.observer else False,
                'scheduler_active': self.is_running
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {'error': str(e)}
    
    def stop(self):
        """Stop all background processes"""
        self.is_running = False
        if self.observer:
            self.observer.stop()
            self.observer.join()
        logger.info("ProductionDocumentManager stopped")


class ProductionHRAssistant:
    """Production-ready HR Assistant with automatic document management"""
    
    def __init__(self, document_directory: str = "../work", chatbot=None):
        self.document_manager = None
        self.chatbot = chatbot
        self.document_directory = document_directory
        self.is_initialized = False
        
    def initialize(self, vectorstore, embeddings):
        """Initialize the production system"""
        logger.info("Initializing Production HR Assistant...")
        
        # Initialize document manager
        self.document_manager = ProductionDocumentManager(
            vectorstore, embeddings, self.document_directory
        )
        
        # Initial document scan
        updated_count = self.document_manager.scan_and_update_all()
        
        # Start background services
        self.document_manager.start_file_watcher()
        self.document_manager.start_scheduler()
        
        self.is_initialized = True
        logger.info("Production HR Assistant initialized successfully")
        
        return updated_count
    
    def get_status_dashboard(self) -> str:
        """Get formatted status dashboard"""
        if not self.document_manager:
            return "❌ Production system not initialized"
            
        stats = self.document_manager.get_system_stats()
        
        dashboard = f"""
🏢 **Nestlé HR Assistant - Production Status**

📊 **System Statistics:**
- Total Documents: {stats.get('total_documents', 0)}
- Total Chunks: {stats.get('total_chunks', 0)}
- Vector Store Size: {stats.get('vectorstore_count', 0)} documents
- Last Update: {stats.get('last_update', 'Never')}

🔄 **Background Services:**
- File Watcher: {'✅ Active' if stats.get('watcher_active') else '❌ Inactive'}
- Scheduler: {'✅ Active' if stats.get('scheduler_active') else '❌ Inactive'}
- Update Queue: {stats.get('queue_size', 0)} pending updates

💡 **Next Actions:**
- Documents are automatically monitored for changes
- Updates processed every 30 minutes
- Full scan every 2 hours
- Daily cleanup at 3:00 AM
        """
        return dashboard.strip()
    
    def force_refresh(self):
        """Force a full document refresh"""
        if not self.document_manager:
            logger.error("Production system not initialized")
            return 0
            
        logger.info("Force refresh requested")
        return self.document_manager.scan_and_update_all()
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down Production HR Assistant...")
        if self.document_manager:
            self.document_manager.stop()


# Usage Example:
"""
# In your notebook, after setting up vectordb and embeddings:

from production_document_manager import ProductionHRAssistant

# Initialize production system
production_system = ProductionHRAssistant(document_directory="../work")
updated_count = production_system.initialize(vectordb, embeddings)

# View status
print(production_system.get_status_dashboard())

# Force refresh if needed
# production_system.force_refresh()

# Shutdown when done
# production_system.shutdown()
"""
