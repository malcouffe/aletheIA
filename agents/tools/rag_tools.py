"""
RAG Tools for Document Retrieval Agents
Handles PDF document search and retrieval using vector databases.
Following smolagents best practices for clear descriptions, error handling, and user guidance.
"""

import os
from smolagents import tool
from ..config.agent_config import RAG_CONFIG


@tool
def search_pdf_documents(query: str, pdf_database_path: str, user_notes: str = "") -> str:
    """
    Search through PDF documents using semantic similarity and return results with detailed source citations.
    
    This tool performs vector similarity search across indexed PDF documents and provides
    comprehensive results with source attribution. Following smolagents best practices
    for detailed error reporting and user guidance.
    
    Args:
        query: The search query to find relevant content in PDF documents.
               Should be specific and descriptive for best results.
        pdf_database_path: Path to the vector database containing indexed PDF documents.
                          Must be a valid path to a ChromaDB database directory.
        user_notes: Additional context or notes about the search to improve relevance.
                   Can include document types, date ranges, or specific topics of interest.
    
    Returns:
        Retrieved document content with detailed source citations including page numbers,
        similarity scores, and document metadata. Or comprehensive error information
        with troubleshooting steps if the search fails.
        
    Usage Examples:
        - search_pdf_documents("AI agents", "/path/to/vector_db")
        - search_pdf_documents("financial metrics", db_path, "Q3 earnings reports")
        - search_pdf_documents("machine learning algorithms", vector_db_path, "technical papers")
    """


    # Input validation
    if not query or not isinstance(query, str):
        error_msg = "❌ ERROR: Query must be a non-empty string"
        print(error_msg)
        return error_msg + "\n\n💡 SOLUTION: Provide a descriptive search query about the content you're looking for"

    if not pdf_database_path or not isinstance(pdf_database_path, str):
        error_msg = "❌ ERROR: 'pdf_database_path' is missing or invalid"
        print(error_msg)
        return error_msg + "\n\n💡 SOLUTION: Provide a valid path to the PDF vector database directory"

    # Check for placeholder path
    if pdf_database_path == "path_to_the_pdf_database":
        placeholder_error = """❌ ERROR: Placeholder path detected! 

🎯 MANAGER AGENT: You must extract the actual database path from additional_args.

✅ CORRECT PATTERN:
1. pdf_context = additional_args.get('pdf_context', {})
2. available_files = pdf_context.get('available_files', [])
3. actual_db_path = available_files[0]['db_path']
4. Use this actual_db_path in the tool call

❌ NEVER use 'path_to_pdf_database' placeholder!

💡 ALTERNATIVE: Use search_pdf_with_context() or search_pdf_from_state() tools instead."""
        print(placeholder_error)
        return placeholder_error

    # Verify database path exists
    if not os.path.exists(pdf_database_path):
        error_msg = f"❌ ERROR: Vector database path not found: {pdf_database_path}"
        print(error_msg)
        return error_msg + "\n\n💡 TROUBLESHOOTING:\n   - Verify the path is correct\n   - Check if PDFs have been indexed\n   - Ensure database directory exists\n   - Contact support if path should be valid"
    
    try:
        print("   📚 Initializing vector database...")
        from langchain_community.vectorstores import Chroma
        from ..core.embedding import get_embedding_function
        
        # Initialize the Chroma vector store dynamically
        vector_store = Chroma(
            persist_directory=pdf_database_path,
            embedding_function=get_embedding_function(),
            collection_name=RAG_CONFIG["collection_name"]
        )
        print("   ✅ Vector database initialized successfully")
    except ImportError as e:
        import_error = f"❌ IMPORT ERROR: Missing required dependencies: {str(e)}"
        print(import_error)
        return import_error + "\n\n💡 SOLUTION: Install required packages: pip install langchain-community chromadb"
    except Exception as e:
        init_error = f"❌ DATABASE ERROR: Failed to initialize vector store from {pdf_database_path}: {str(e)}"
        print(init_error)
        return init_error + "\n\n💡 TROUBLESHOOTING:\n   - Check if database is corrupted\n   - Verify ChromaDB format compatibility\n   - Try rebuilding the database\n   - Ensure embedding function is available"

    try:
        print(f"   🔍 Performing similarity search for: '{query}'")
        
        # Perform the similarity search with metadata
        search_k = RAG_CONFIG["similarity_search_k"]
        results = vector_store.similarity_search_with_score(query, k=search_k)
        
        print(f"   📋 Found {len(results)} relevant documents")
        
        if not results:
            no_results = f"""📚 NO RELEVANT DOCUMENTS FOUND

🔍 SEARCH DETAILS:
   Query: "{query}"
   Database: {pdf_database_path}
   Search parameters: Top {search_k} results
   User notes: {user_notes if user_notes else 'None provided'}

💡 SUGGESTIONS TO IMPROVE SEARCH:
   - Try broader or different keywords
   - Check if relevant documents are indexed
   - Verify the database contains the expected content
   - Use synonyms or related terms
   - Try shorter, more specific queries

🛠️ TECHNICAL CHECKS:
   - Ensure PDFs were properly processed and indexed
   - Verify embedding function is working correctly
   - Check if database collection exists: {RAG_CONFIG["collection_name"]}"""
            print("   ❌ No relevant documents found")
            return no_results

        # Format results with rich metadata
        formatted_results = []
        formatted_results.append("📚 PDF DOCUMENT SEARCH RESULTS")
        formatted_results.append("=" * 60)
        formatted_results.append(f"🔍 Query: {query}")
        formatted_results.append(f"📁 Database: {pdf_database_path}")
        formatted_results.append(f"📊 Results: {len(results)} documents found")
        if user_notes:
            formatted_results.append(f"📝 Context: {user_notes}")
        formatted_results.append("")

        total_content_length = 0
        for i, (doc, score) in enumerate(results, 1):
        # Extract metadata
            metadata = doc.metadata or {}
            source = metadata.get('source', 'Unknown document')
            page = metadata.get('page', 'Unknown page')
        
            # Format similarity score as percentage
            similarity_percent = (1 - score) * 100 if score <= 1 else 0
            
            # Prepare document content
            content = doc.page_content.strip()
            content_length = len(content)
            total_content_length += content_length
            
            # Add result entry
            formatted_results.append(f"📄 RESULT #{i} (Relevance: {similarity_percent:.1f}%)")
            formatted_results.append(f"   📁 Source: {source}")
            formatted_results.append(f"   📖 Page: {page}")
            formatted_results.append(f"   📏 Length: {content_length} characters")
            formatted_results.append(f"   🎯 Similarity Score: {1-score:.3f}" if score <= 1 else f"   🎯 Distance: {score:.3f}")
            formatted_results.append("")
            formatted_results.append("📖 CONTENT:")
            formatted_results.append("-" * 40)
            formatted_results.append(content)
            formatted_results.append("")
            formatted_results.append("-" * 60)
            formatted_results.append("")
            
            print(f"   📄 Result {i}: {similarity_percent:.1f}% relevance, {content_length} chars")
    
        # Add summary
        formatted_results.append("📊 SEARCH SUMMARY:")
        formatted_results.append(f"   ✅ Total documents retrieved: {len(results)}")
        formatted_results.append(f"   📏 Total content length: {total_content_length} characters")
        formatted_results.append(f"   🎯 Average relevance: {sum((1-score)*100 for _, score in results if score <= 1)/len(results):.1f}%")
        formatted_results.append("")
        formatted_results.append("💡 TO USE THESE RESULTS:")
        formatted_results.append("   - Review each document for relevant information")
        formatted_results.append("   - Check page numbers for original source verification")
        formatted_results.append("   - Use higher relevance results for primary analysis")
        formatted_results.append("   - Consider refining query if results aren't specific enough")

        result_text = "\n".join(formatted_results)
        print(f"✅ PDF search completed successfully: {len(results)} results, {total_content_length} total chars")
        return result_text

    except Exception as e:
        search_error = f"❌ SEARCH ERROR: Failed to search PDF database: {str(e)}"
        print(search_error)
        return search_error + f"\n\n💡 TROUBLESHOOTING:\n   - Check if query is properly formatted\n   - Verify database is not corrupted\n   - Ensure sufficient memory for search operation\n   - Try with a simpler query first\n   - Contact support if error persists"


@tool  
def search_pdf_with_context(user_query: str, pdf_context_dict: dict = None) -> str:
    """
    Search PDF documents using provided context dictionary from agent state.
    
    This tool is designed to work with smolagents' additional_args mechanism where
    context information becomes available as state variables. Following smolagents
    best practices for state variable handling and comprehensive error reporting.
    
    Args:
        user_query: The user's query about PDF documents. Should be specific and
                   descriptive for optimal search results.
        pdf_context_dict: Dictionary containing PDF context with available_files.
                         Usually comes from the pdf_context state variable passed
                         via additional_args in agent.run().
        
    Returns:
        Comprehensive search results from PDF documents with source citations,
        relevance scores, and metadata. Or detailed error message with guidance
        if context is missing or invalid.
        
    Usage Examples:
        - search_pdf_with_context("machine learning models", pdf_context)
        - search_pdf_with_context("financial analysis", context_from_state)
        - search_pdf_with_context("research methodology", pdf_context_dict)
    """

    
    # Input validation
    if not user_query or not isinstance(user_query, str):
        error_msg = "❌ ERROR: User query must be a non-empty string"
        print(error_msg)
        return error_msg + "\n\n💡 SOLUTION: Provide a descriptive search query"
    
    if not pdf_context_dict:
        no_context_msg = """❌ NO PDF CONTEXT PROVIDED

🎯 CONTEXT REQUIRED: This tool needs PDF context to function properly.

✅ TYPICAL USAGE PATTERN:
```python
# In agent code:
if 'pdf_context' in locals() and pdf_context:
    result = search_pdf_with_context(user_query, pdf_context)
else:
    # Fallback to web search or other tools
    result = "No PDF context available. Please upload PDF files first."
```

💡 ALTERNATIVES:
   - Use search_agent for web research instead
   - Upload and index PDF files first
   - Use enhanced_visit_webpage for online research
   - Contact support if PDFs should be available"""
        print(no_context_msg)
        return no_context_msg
    
    if not isinstance(pdf_context_dict, dict):
        type_error = f"❌ ERROR: PDF context must be a dictionary, received: {type(pdf_context_dict)}"
        print(type_error)
        return type_error + "\n\n💡 EXPECTED FORMAT: {'available_files': [...], 'count': N, 'metadata': {...}}"
    
    available_files = pdf_context_dict.get("available_files", [])
    print(f"📚 Available PDF files in context: {len(available_files)}")
    
    if not available_files:
        no_files_msg = """❌ NO PDF FILES IN CONTEXT

📋 CONTEXT STRUCTURE RECEIVED:
""" + str(pdf_context_dict) + """

💡 EXPECTED STRUCTURE:
{
    "available_files": [
        {
            "filename": "document.pdf",
            "db_path": "/path/to/vector_db",
            "user_notes": "description",
            "classification": "category"
        }
    ],
    "count": 1
}

🛠️ SOLUTIONS:
   - Upload and index PDF files first
   - Verify PDF processing completed successfully
   - Check if context is properly prepared
   - Use web search tools as alternative"""
        print(no_files_msg)
        return no_files_msg
    
    # Get the database path from the first available file
    first_file = available_files[0]
    db_path = first_file.get("db_path", "")
    user_notes = first_file.get("user_notes", "")
    classification = first_file.get("classification", "General")
    filename = first_file.get("filename", "Unknown PDF")
    
    print(f"📄 Using PDF: {filename}")
    print(f"📁 Database path: {db_path}")
    print(f"🏷️ Classification: {classification}")
    
    if not db_path or db_path == "path_to_pdf_database":
        path_error = """❌ ERROR: No valid PDF database path found in context

📋 FILE CONTEXT RECEIVED:
""" + str(first_file) + """

💡 TROUBLESHOOTING:
   - PDF indexing may have failed
   - Database path not properly set during upload
   - Context preparation error
   - Try re-uploading and indexing the PDF files"""
        print(path_error)
        return path_error
    
    # Search the PDF database with the actual path

    search_notes = f"{user_notes} - Classification: {classification}" if user_notes else f"Classification: {classification}"
    
    try:
        result = search_pdf_documents(user_query, db_path, search_notes)
        print("✅ PDF search completed successfully")
        
        # Enhance result with context information
        context_header = f"""🔍 PDF SEARCH WITH CONTEXT

    📄 Document: {filename}
    🏷️ Classification: {classification}
    📝 Notes: {user_notes if user_notes else 'None provided'}
    📊 Available files: {len(available_files)}

    {'='*60}

    """
        return context_header + result
        
    except Exception as e:
        search_error = f"❌ SEARCH ERROR with context: {str(e)}"
        print(search_error)
        return search_error + f"\n\n💡 CONTEXT INFO:\n   - File: {filename}\n   - DB Path: {db_path}\n   - Classification: {classification}"


@tool
def search_pdf_from_state(user_query: str) -> str:
    """
    Search PDF documents by accessing PDF context from agent state variables.
    
    This tool demonstrates the smolagents additional_args mechanism where context
    becomes available as state variables like 'pdf_context'. Following smolagents
    best practices for state variable access patterns and informative error handling.
    
    The agent calling this tool should check for pdf_context state variable:
    ```python
    if 'pdf_context' in locals() and pdf_context:
        # pdf_context is available, use search_pdf_with_context instead
        result = search_pdf_with_context(user_query, pdf_context)
    else:
        # Fallback behavior or guidance
        result = search_pdf_from_state(user_query)
    ```
    
    Args:
        user_query: The user's query about PDF documents. Should be descriptive
                   and specific for best search results.
        
    Returns:
        Guidance for the agent on how to access PDF context from state variables,
        or search results if context access patterns are properly implemented.
        
    Usage Examples:
        - search_pdf_from_state("AI research findings")
        - search_pdf_from_state("financial performance metrics")
        - search_pdf_from_state("technical specifications")
    """
    # Input validation
    if not user_query or not isinstance(user_query, str):
        error_msg = "❌ ERROR: User query must be a non-empty string"
        print(error_msg)
        return error_msg + "\n\n💡 SOLUTION: Provide a descriptive search query"
    
    # This tool primarily provides guidance on accessing state variables
    guidance_message = f"""
🎯 PDF SEARCH FROM STATE VARIABLES

📝 QUERY: "{user_query}"

🔍 SMOLAGENTS STATE VARIABLE ACCESS PATTERN:

The agent should use this pattern to access PDF context:

```python
# Check if PDF context is available as state variable
if 'pdf_context' in locals() and pdf_context:
    print(f"PDF context available: {{pdf_context['count']}} files")
    # Use the context-aware search tool
    result = search_pdf_with_context("{user_query}", pdf_context)
    final_answer(result)
else:
    print("No PDF context available in state variables")
    # Provide alternative or guidance
    final_answer("No PDF documents available. Please upload PDF files first or use web search tools for online research.")
```

💡 ALTERNATIVE APPROACHES:
1. **Context-aware search**: Use search_pdf_with_context(query, pdf_context)
2. **Direct database access**: Use search_pdf_documents(query, db_path)
3. **Web research**: Use enhanced_visit_webpage() for online sources
4. **Combined approach**: Try PDF first, then fallback to web search

🛠️ TROUBLESHOOTING:
- Verify additional_args contains pdf_context
- Check if PDFs were properly uploaded and indexed
- Ensure agent state variables are accessible
- Use context access tools for debugging

📚 SMOLAGENTS DOCUMENTATION:
According to smolagents best practices, additional_args passed to agent.run()
become available as state variables in the agent's execution environment.
"""
    
    print("💡 Providing state variable access guidance")
    return guidance_message


@tool
def extract_pdf_context_and_delegate(user_query: str) -> str:
    """
    Helper tool that provides instructions for extracting PDF context from additional_args
    and formatting the correct delegation pattern for RAG operations.
    
    This tool demonstrates smolagents best practices for additional_args handling
    where they become state variables accessible in the agent's Python execution environment.
    Following comprehensive documentation and user guidance principles.
    
    Args:
        user_query: The user's query about PDF documents that needs to be processed
                   with proper context extraction and delegation.
        
    Returns:
        Detailed instructions for the agent on how to access PDF context from state
        variables and properly delegate to RAG tools with comprehensive examples.
        
    Usage Examples:
        - extract_pdf_context_and_delegate("research findings")
        - extract_pdf_context_and_delegate("financial analysis")
        - extract_pdf_context_and_delegate("technical documentation")
    """

    
    # Input validation
    if not user_query or not isinstance(user_query, str):
        error_msg = "❌ ERROR: User query must be a non-empty string"
        print(error_msg)
        return error_msg + "\n\n💡 SOLUTION: Provide a query about PDF documents"
    
    # According to smolagents documentation, additional_args are added to agent.state
    # and become accessible as variables in Python code generation
    
    instructions = f"""
🔍 PDF CONTEXT EXTRACTION AND DELEGATION

📝 QUERY TO PROCESS: "{user_query}"

🎯 SMOLAGENTS CONTEXT ACCESS PATTERN:

According to smolagents documentation, additional_args passed to agent.run()
become available as state variables in the agent's Python execution environment.

```python
# STEP 1: Check if PDF context is available
if 'pdf_context' in locals() and pdf_context:
    print(f"✅ PDF context found: {{pdf_context.get('count', 0)}} files available")
    
    # STEP 2: Extract context information
    available_files = pdf_context.get('available_files', [])
    if available_files:
        first_file = available_files[0]
        filename = first_file.get('filename', 'Unknown')
        classification = first_file.get('classification', 'General')
        print(f"📄 Primary document: {{filename}} ({{classification}})")
    
    # STEP 3: Use context-aware search
    result = search_pdf_with_context("{user_query}", pdf_context)
    print("🔍 PDF search completed")
    final_answer(result)
    
elif 'csv_context' in locals() and csv_context:
    print("📊 CSV context available, but query seems PDF-related")
    final_answer("Query appears to be about PDF documents, but only CSV data is available. Please clarify if you want to analyze the CSV data instead.")
    
else:
    print("❌ No document context available")
    final_answer("No PDF documents are currently available. Please upload PDF files first, or I can help you search for information online using web search tools.")
```

🔄 ALTERNATIVE DELEGATION PATTERNS:

**Pattern 1: Direct context passing**
```python
if 'pdf_context' in locals() and pdf_context:
    search_result = search_pdf_with_context("{user_query}", pdf_context)
    final_answer(search_result)
```

**Pattern 2: Database path extraction**
```python
if 'pdf_context' in locals() and pdf_context:
    files = pdf_context.get('available_files', [])
    if files:
        db_path = files[0].get('db_path')
        notes = files[0].get('user_notes', '')
        result = search_pdf_documents("{user_query}", db_path, notes)
        final_answer(result)
```

**Pattern 3: Fallback to web search**
```python
if 'pdf_context' in locals() and pdf_context:
    result = search_pdf_with_context("{user_query}", pdf_context)
else:
    print("No PDF context, using web search")
    result = enhanced_visit_webpage("https://search.relevant.site")
final_answer(result)
```

💡 TROUBLESHOOTING GUIDE:

🔍 **Context Not Available**:
- Check if additional_args contains pdf_context
- Verify PDFs were uploaded and indexed
- Use check_context_availability() tool for debugging

📚 **Empty Context**:
- Ensure PDF processing completed successfully
- Verify database paths are valid
- Check if files were properly indexed

🛠️ **Tool Selection**:
- Use search_pdf_with_context() for state variable context
- Use search_pdf_documents() for direct database access
- Use search_pdf_from_state() for guidance and patterns

📖 **Best Practices**:
- Always check context availability first
- Provide meaningful fallback options
- Log context information for debugging
- Handle errors gracefully with user guidance
"""
    
    print("✅ Context extraction instructions provided")
    return instructions


@tool
def get_citation_help() -> str:
    """
    Provide comprehensive guidance on citing PDF sources retrieved through RAG operations.
    
    This tool offers detailed instructions for properly attributing and citing information
    found through PDF document searches, following academic and professional standards.
    Aligned with smolagents best practices for helpful, informative tools.
    
    Returns:
        Comprehensive guide on citation formats, source attribution, and best practices
        for referencing PDF documents found through vector database searches.
        
    Usage Examples:
        - get_citation_help()  # Get general citation guidance
        - Use after search_pdf_documents() to learn proper attribution
        - Reference when preparing reports or analyses with PDF sources
    """
    print("📚 get_citation_help: Providing comprehensive citation guidance")
    
    citation_guide = """
📚 COMPREHENSIVE PDF CITATION GUIDE

🎯 PURPOSE: How to properly cite PDF documents found through RAG searches

📋 CITATION FORMATS BY TYPE:

📄 **Academic Papers**:
Author, A. A. (Year). Title of paper. Journal Name, Volume(Issue), pages.
Example: Smith, J. (2023). AI in Healthcare. Tech Review, 15(3), 45-67.

📊 **Reports & Whitepapers**:
Organization. (Year). Report Title. Publisher.
Example: McKinsey Institute. (2023). Future of Work Report. McKinsey & Company.

📖 **Books & eBooks**:
Author, A. A. (Year). Book Title. Publisher.
Example: Johnson, M. (2023). Machine Learning Fundamentals. Tech Publications.

🏢 **Corporate Documents**:
Company Name. (Year). Document Title. Document Type.
Example: Apple Inc. (2023). Annual Report 2023. SEC Filing 10-K.

🔍 **INFORMATION FROM RAG SEARCHES**:

When citing content found through PDF searches, include:

✅ **Required Elements**:
- Original document title and author
- Page number where information was found
- Date of document
- Source of the PDF (if different from original)
- Date accessed (for digital sources)

📝 **Citation Template for RAG Results**:
"According to [Author/Organization] in [Document Title] (page [X]), [quote or paraphrase]. Retrieved from [source] on [date]."

💡 **Example RAG Citation**:
"The study found that 'AI adoption increased by 40% in healthcare' (Chen et al., Healthcare AI Review, page 23). Retrieved from company database on March 15, 2024."

🛡️ **ATTRIBUTION BEST PRACTICES**:

1. **Always include page numbers** from search results
2. **Quote directly** when possible, with quotation marks
3. **Paraphrase responsibly** and indicate the source
4. **Note similarity scores** if relevant to credibility
5. **Include retrieval date** for digital documents

⚠️ **IMPORTANT CONSIDERATIONS**:

🔍 **Verify Accuracy**:
- Cross-reference with original source when possible
- Note that RAG results are excerpts, not complete context
- Consider similarity scores as relevance indicators

📊 **Multiple Sources**:
- Cite each PDF source separately
- Note when information appears in multiple documents
- Prioritize higher similarity scores for primary citations

🎯 **Professional Reports**:
- Use footnotes or endnotes for detailed source information
- Include a bibliography of all PDF sources consulted
- Note the search methodology used

📝 **SAMPLE CITATIONS**:

**Academic Style (APA)**:
Johnson, M. (2023). "Machine learning applications in finance" (p. 156). In Financial Technology Review. Boston: Tech Press. Retrieved via document search March 2024.

**Business Report Style**:
As noted in the Q3 Financial Report (Apple Inc., 2023, p. 45), revenue increased by 15% year-over-year. Document accessed via corporate database, March 15, 2024.

**Legal/Compliance Style**:
Reference: SEC Filing 10-K, Microsoft Corporation, Filed December 31, 2023, Page 67, Section 3.2. Retrieved through regulatory database search on March 2024.

🔄 **WORKFLOW INTEGRATION**:

1. **During Search**: Note source metadata from results
2. **When Quoting**: Copy exact text with page references
3. **In Writing**: Use proper attribution format
4. **Final Review**: Verify all sources are properly cited

📚 **ADDITIONAL RESOURCES**:

- Check institutional style guides (APA, MLA, Chicago)
- Verify citation requirements for your specific use case
- Consider legal implications of document usage
- Maintain source documentation for future reference

⚖️ **LEGAL & ETHICAL NOTES**:

- Respect copyright and fair use principles
- Ensure you have rights to access and cite documents
- Note confidentiality levels of internal documents
- Follow organizational policies for document citation
"""
    
    print("✅ Citation guidance provided successfully")
    return citation_guide 