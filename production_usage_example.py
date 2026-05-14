"""
Production Usage Example for HR Assistant
Copy this code into a new cell in your Jupyter notebook
"""

# Install additional required packages for production features
# Run this in your notebook:
# %pip install watchdog schedule

# Production Usage - Add this to your notebook after the basic setup:

"""
# Cell: Production System Integration
import sys
sys.path.append('.')  # Add current directory to path

from production_document_manager import ProductionHRAssistant

# Initialize production system
production_system = ProductionHRAssistant(
    document_directory="../work",  # Your document directory
    chatbot=hr_chatbot  # Your existing chatbot
)

# Initialize with your existing vector store and embeddings
updated_count = production_system.initialize(vectordb, embeddings)

print(f"🚀 Production system initialized with {updated_count} updated documents")
print("=" * 60)
print(production_system.get_status_dashboard())

# The system now automatically:
# 1. Monitors document files for changes
# 2. Updates vector store when files change
# 3. Runs periodic scans every 2 hours
# 4. Cleans up deleted files daily at 3:00 AM
# 5. Queues updates for batch processing

# Manual operations (if needed):
# Force refresh all documents:
# refresh_count = production_system.force_refresh()

# Get current statistics:
# stats = production_system.document_manager.get_system_stats()
# print(f"Current stats: {stats}")

# Shutdown gracefully (when done):
# production_system.shutdown()
"""

# Enhanced Gradio Interface with Production Features
"""
# Cell: Enhanced Interface with Production Status
import gradio as gr

def enhanced_chatbot_interface(message, history):
    """Enhanced chatbot with production monitoring"""
    try:
        response = hr_chatbot.chat(message)
        
        # Add production status indicator
        if production_system.is_initialized:
            stats = production_system.document_manager.get_system_stats()
            status_emoji = "🟢" if stats.get('watcher_active') else "🔴"
            response += f"\n\n{status_emoji} *System monitoring active - {stats.get('total_documents', 0)} documents indexed*"
        
        return response
    except Exception as e:
        return f"Error: {e}"

def get_system_status():
    """Get formatted system status for display"""
    if production_system.is_initialized:
        return production_system.get_status_dashboard()
    else:
        return "❌ Production system not initialized"

# Create enhanced interface
with gr.Blocks(theme=gr.themes.Soft()) as enhanced_demo:
    gr.Markdown("# 🏢 Nestlé HR Assistant - Production Edition")
    
    with gr.Tab("💬 Chat"):
        chatbot = gr.ChatInterface(
            fn=enhanced_chatbot_interface,
            title="HR Assistant Chat",
            examples=[
                "What is the company's policy on remote work?",
                "How do I request time off?",
                "What are the performance review procedures?"
            ]
        )
    
    with gr.Tab("📊 System Status"):
        status_display = gr.Markdown(get_system_status())
        refresh_btn = gr.Button("🔄 Refresh Status")
        force_refresh_btn = gr.Button("🔁 Force Document Refresh")
        
        refresh_btn.click(get_system_status, outputs=status_display)
        force_refresh_btn.click(
            lambda: f"Refreshed! Updated {production_system.force_refresh()} documents.",
            outputs=status_display
        )

# Launch enhanced interface
enhanced_demo.launch(share=False, server_port=7861)
"""

# Monitoring and Alerting Setup
"""
# Cell: Monitoring and Alerting
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

class ProductionMonitor:
    def __init__(self, production_system):
        self.production_system = production_system
        self.alert_thresholds = {
            'max_queue_size': 10,
            'min_vectorstore_count': 1,
            'max_hours_since_update': 24
        }
    
    def check_system_health(self):
        """Check system health and return status"""
        stats = self.production_system.document_manager.get_system_stats()
        
        issues = []
        
        # Check queue size
        if stats.get('queue_size', 0) > self.alert_thresholds['max_queue_size']:
            issues.append(f"High update queue: {stats['queue_size']} items")
        
        # Check vector store
        if stats.get('vectorstore_count', 0) < self.alert_thresholds['min_vectorstore_count']:
            issues.append("Vector store appears empty")
        
        # Check last update time
        last_update = stats.get('last_update', 'Never')
        if last_update != 'Never':
            try:
                update_time = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                hours_since = (datetime.now(update_time.tzinfo) - update_time).total_seconds() / 3600
                if hours_since > self.alert_thresholds['max_hours_since_update']:
                    issues.append(f"No updates for {hours_since:.1f} hours")
            except:
                issues.append("Cannot parse last update time")
        
        return {
            'healthy': len(issues) == 0,
            'issues': issues,
            'stats': stats
        }
    
    def send_alert(self, message):
        """Send alert email (configure with your SMTP settings)"""
        # Configure your email settings
        # smtp_server = "smtp.gmail.com"
        # smtp_port = 587
        # sender_email = "your_email@gmail.com"
        # sender_password = "your_app_password"
        # recipient_email = "admin@company.com"
        
        # msg = MIMEText(f"HR Assistant Alert: {message}")
        # msg['Subject'] = 'HR Assistant System Alert'
        # msg['From'] = sender_email
        # msg['To'] = recipient_email
        
        # try:
        #     server = smtplib.SMTP(smtp_server, smtp_port)
        #     server.starttls()
        #     server.login(sender_email, sender_password)
        #     server.send_message(msg)
        #     server.quit()
        #     logger.info("Alert sent successfully")
        # except Exception as e:
        #     logger.error(f"Failed to send alert: {e}")
        pass

# Initialize monitor
monitor = ProductionMonitor(production_system)

# Periodic health check
def periodic_health_check():
    health = monitor.check_system_health()
    if not health['healthy']:
        alert_message = "Issues detected: " + "; ".join(health['issues'])
        logger.warning(f"System health issues: {alert_message}")
        # monitor.send_alert(alert_message)
    return health

# Schedule health checks
import schedule
schedule.every(1).hours.do(periodic_health_check)

print("✅ Production monitoring initialized")
print("📊 Health checks will run every hour")
"""
