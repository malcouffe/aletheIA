"""
Chat interface functionality for handling user interactions.
"""
import streamlit as st
import time
import datetime


def display_chat_interface(model, agent_manager):
    """Display the chat interface and handle user interactions."""
    # Display existing messages with enhanced formatting
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Check if this is a system/notification message
            is_system_message = _is_system_message(message["content"])
            
            if message["role"] == "assistant":
                if is_system_message:
                    # Special formatting for system messages (file uploads, indexing, etc.)
                    _display_system_message(message["content"], message.get("timestamp"))
                else:
                    # Regular assistant response
                    st.markdown(message["content"])
                    _display_timestamp(message.get("timestamp"))
            else:
                # User message
                st.markdown(message["content"])
                _display_timestamp(message.get("timestamp"))

    # Handle new user input
    if prompt := st.chat_input("Quel est votre question?"):
        # Add timestamp to user message
        user_message = {
            "role": "user", 
            "content": prompt,
            "timestamp": time.time()
        }
        st.session_state.messages.append(user_message)
        
        with st.chat_message("user"):
            st.markdown(prompt)
            _display_timestamp(user_message["timestamp"])

        with st.chat_message("assistant"):
            # Create a container for the response that can be updated
            response_container = st.container()
            
            with response_container:
                status_placeholder = st.empty()
                status_placeholder.markdown("🔄 **Traitement en cours...**")
                
                # Process the query and get response
                final_response = _process_user_query(
                    prompt, model, agent_manager,
                    response_container  # Pass container for real-time updates
                )
                
                # Clear the status and show final response
                status_placeholder.empty()
                
                # Format and display the final response
                formatted_response = _format_agent_response(final_response)
                st.markdown(formatted_response)
                
                # Store the formatted response in session state with timestamp
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": formatted_response,
                    "timestamp": time.time()
                })


def _is_system_message(content):
    """Check if a message is a system/notification message."""
    system_indicators = [
        "📤", "✅", "🔄", "❌", "⚠️", "ℹ️", "🚀", "🎉", "💥",
        "**Début du traitement**", "**traité avec succès**", "**Démarrage de l'indexation**",
        "**Indexation terminée**", "**Configuration en cours**", "**Erreur"
    ]
    return any(indicator in content for indicator in system_indicators)


def _display_system_message(content, timestamp=None):
    """Display system messages with special formatting."""
    # Create a bordered container for system messages
    with st.container():
        # Use info/success/error styling based on content
        if "✅" in content or "🎉" in content or "traité avec succès" in content:
            st.success(content)
        elif "❌" in content or "💥" in content or "Erreur" in content:
            st.error(content)
        elif "⚠️" in content:
            st.warning(content)
        elif "🔄" in content or "en cours" in content:
            st.info(content)
        else:
            st.info(content)
    
    _display_timestamp(timestamp)


def _display_timestamp(timestamp):
    """Display a timestamp for messages."""
    if timestamp:
        try:
            dt = datetime.datetime.fromtimestamp(timestamp)
            time_str = dt.strftime("%H:%M:%S")
            st.caption(f"🕒 {time_str}")
        except (ValueError, TypeError):
            pass


def _process_user_query(prompt, model, agent_manager, response_container=None):
    """Process the user query and return the response using the new clean architecture."""
    if not model or not agent_manager:
        return "Désolé, les agents IA ne sont pas correctement initialisés. Vérifiez la configuration de la clé API."
    
    try:
        available_pdfs_context, available_csvs_context = _prepare_context()
        
        # Show progress if we have a container
        if response_container:
            with response_container:
                progress_placeholder = st.empty()
                progress_placeholder.markdown("🤖 **Agent en cours d'exécution...**")
        
        # Use the new clean AgentManager interface
        final_response = agent_manager.run_query(
            prompt,  # user query
            available_pdfs_context,
            available_csvs_context
        )
        
        # Clear progress indicator
        if response_container:
            progress_placeholder.empty()
        
        # Convert non-string responses to string
        if not isinstance(final_response, str):
            final_response = str(final_response)
            
        return final_response
        
    except Exception as e:
        error_response = f"❌ **Erreur lors de l'exécution de l'agent:**\n\n```\n{str(e)}\n```"
        if response_container:
            st.error(f"Erreur: {str(e)}")
        return error_response


def _format_agent_response(response):
    """Format the agent response for better display - simplified approach."""
    if not response:
        return "Aucune réponse générée."
    
    # Convert to string and basic cleanup
    formatted_response = str(response).strip()
    
    # Remove common wrapper patterns
    if formatted_response.startswith("final_answer(") and formatted_response.endswith(")"):
        # Extract content from final_answer() wrapper
        start_quote = formatted_response.find('"') + 1
        end_quote = formatted_response.rfind('"')
        if start_quote > 0 and end_quote > start_quote:
            formatted_response = formatted_response[start_quote:end_quote]
    
    # Basic cleanup - trust the LLM to provide natural language
    formatted_response = formatted_response.replace('\\n', '\n')
    formatted_response = formatted_response.replace('\\"', '"')
    
    return formatted_response if formatted_response else "Aucune réponse générée."


def _prepare_context():
    """Prepare PDF and CSV context for the manager agent."""
    available_pdfs_context = []
    available_csvs_context = []
    
    for fid, details in st.session_state.get('processed_files', {}).items():
        if details.get('type') == 'pdf' and details.get('indexed') and details.get('db_path'):
            available_pdfs_context.append({
                'file_id': fid,
                'filename': details.get('filename', 'Unknown PDF'),
                'classification': details.get('classification'),
                'db_path': details.get('db_path'),
                'user_notes': details.get('user_notes', ''),
                'summary': details.get('summary', '')
            })
        elif details.get('type') == 'csv' and details.get('status') == 'ready':
            csv_args = details.get('csv_args', {})
            if not isinstance(csv_args, dict): 
                csv_args = {}
            
            csv_context = {
                'file_id': fid,
                'filename': details.get('filename', 'Unknown CSV'),
                'csv_args': csv_args,
                'user_notes': details.get('user_notes', '')
            }
            
            if 'rows' in details:
                csv_context['csv_args']['rows'] = details['rows']
            if 'columns' in details:
                csv_context['csv_args']['columns'] = details['columns']
                
            available_csvs_context.append(csv_context)
    
    return available_pdfs_context, available_csvs_context 