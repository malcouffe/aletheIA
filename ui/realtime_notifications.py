"""
Real-time notification system for status updates.
"""
import streamlit as st
import time
import threading


def create_status_toast(message: str, status: str = "info"):
    """Create a toast notification for status updates."""
    toast_container = st.empty()
    
    if status == "success":
        toast_container.success(message)
    elif status == "error":
        toast_container.error(message)
    elif status == "warning":
        toast_container.warning(message)
    else:
        toast_container.info(message)
    
    # Auto-clear after 3 seconds
    def clear_toast():
        time.sleep(3)
        toast_container.empty()
    
    threading.Thread(target=clear_toast, daemon=True).start()


# Helper functions for common notifications
def notify_indexing_started(filename: str):
    """Show notification when indexing starts."""
    create_status_toast(f"🔄 Indexation de '{filename}' démarrée", "info")


def notify_indexing_completed(filename: str):
    """Show notification when indexing completes."""
    create_status_toast(f"✅ '{filename}' indexé avec succès!", "success")


def notify_indexing_failed(filename: str, error: str = ""):
    """Show notification when indexing fails."""
    message = f"❌ Échec de l'indexation de '{filename}'"
    if error:
        message += f": {error}"
    create_status_toast(message, "error") 