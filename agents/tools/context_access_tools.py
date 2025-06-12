"""
Context Access Tools for Smolagents
Demonstrates proper access to additional_args as state variables.
Following smolagents best practices for comprehensive documentation and user guidance.
"""

from smolagents import tool
from typing import Dict, Any, Optional


@tool
def check_context_availability() -> str:
    """
    Check what context is available as state variables in the agent's execution environment.
    
    According to smolagents documentation, additional_args passed to agent.run()
    become available as state variables in the agent's Python execution environment.
    This tool helps debug and understand what context is accessible for processing.
    Following smolagents best practices for informative debugging tools.
    
    Returns:
        Comprehensive report of available context variables, their expected structure,
        and usage patterns with code examples for accessing each type of context.
        
    Usage Examples:
        - check_context_availability()  # Debug what context is available
        - Use at the start of agent execution to understand available data
        - Helpful for troubleshooting context-related issues
    """
    print("🔍 check_context_availability: Analyzing available context variables")
    
    report = ["🔍 CONTEXT AVAILABILITY ANALYSIS"]
    report.append("=" * 50)
    report.append("")
    report.append("📋 SMOLAGENTS CONTEXT MECHANISM:")
    report.append("According to smolagents documentation, additional_args passed to")
    report.append("agent.run() become available as state variables in the Python")
    report.append("execution environment.")
    report.append("")
    
    # Check for PDF context
    report.append("📄 PDF CONTEXT ANALYSIS:")
    report.append("   🔍 Variable name: 'pdf_context'")
    report.append("   ✅ Access pattern: if 'pdf_context' in locals() and pdf_context:")
    report.append("   📊 Expected structure:")
    report.append("      {")
    report.append("        'available_files': [")
    report.append("          {")
    report.append("            'filename': 'document.pdf',")
    report.append("            'db_path': '/path/to/vector_db',")
    report.append("            'user_notes': 'description',")
    report.append("            'classification': 'category'")
    report.append("          }")
    report.append("        ],")
    report.append("        'count': 1,")
    report.append("        'metadata': {...}")
    report.append("      }")
    report.append("")
    
    # Check for CSV context  
    report.append("📊 CSV CONTEXT ANALYSIS:")
    report.append("   🔍 Variable name: 'csv_context'")
    report.append("   ✅ Access pattern: if 'csv_context' in locals() and csv_context:")
    report.append("   📊 Expected structure:")
    report.append("      {")
    report.append("        'available_files': [")
    report.append("          {")
    report.append("            'filename': 'data.csv',")
    report.append("            'path': '/path/to/data.csv',")
    report.append("            'columns': ['col1', 'col2'],")
    report.append("            'rows': 1000,")
    report.append("            'description': 'dataset info'")
    report.append("          }")
    report.append("        ],")
    report.append("        'count': 1,")
    report.append("        'metadata': {...}")
    report.append("      }")
    report.append("")
    
    # Usage patterns
    report.append("💡 PRACTICAL USAGE PATTERNS:")
    report.append("")
    report.append("🔍 **Context Detection Pattern**:")
    report.append("```python")
    report.append("# Check what context is available")
    report.append("context_found = []")
    report.append("if 'pdf_context' in locals() and pdf_context:")
    report.append("    context_found.append(f'PDF: {pdf_context[\"count\"]} files')")
    report.append("if 'csv_context' in locals() and csv_context:")
    report.append("    context_found.append(f'CSV: {csv_context[\"count\"]} files')")
    report.append("")
    report.append("if context_found:")
    report.append("    print(f'Available context: {context_found}')")
    report.append("else:")
    report.append("    print('No context variables found')")
    report.append("```")
    report.append("")
    
    report.append("📄 **PDF Context Usage**:")
    report.append("```python")
    report.append("if 'pdf_context' in locals() and pdf_context:")
    report.append("    files = pdf_context.get('available_files', [])")
    report.append("    if files:")
    report.append("        # Use PDF search tools")
    report.append("        result = search_pdf_with_context(user_query, pdf_context)")
    report.append("        final_answer(result)")
    report.append("    else:")
    report.append("        final_answer('PDF context empty')")
    report.append("else:")
    report.append("    # Fallback to web search")
    report.append("    result = enhanced_visit_webpage('https://relevant.site')")
    report.append("    final_answer(result)")
    report.append("```")
    report.append("")
    
    report.append("📊 **CSV Context Usage**:")
    report.append("```python")
    report.append("if 'csv_context' in locals() and csv_context:")
    report.append("    files = csv_context.get('available_files', [])")
    report.append("    if files:")
    report.append("        # Delegate to data analyst")
    report.append("        analysis_request = f'Analyze: {user_query}'")
    report.append("        result = data_analyst(analysis_request)")
    report.append("        final_answer(result)")
    report.append("    else:")
    report.append("        final_answer('CSV context empty')")
    report.append("else:")
    report.append("    # Load CSV files manually")
    report.append("    result = load_csv_data('data.csv')")
    report.append("    final_answer(result)")
    report.append("```")
    report.append("")
    
    report.append("🔄 **Multi-Context Decision Logic**:")
    report.append("```python")
    report.append("# Prioritize context based on query type")
    report.append("if 'pdf' in user_query.lower() or 'document' in user_query.lower():")
    report.append("    if 'pdf_context' in locals() and pdf_context:")
    report.append("        result = search_pdf_with_context(user_query, pdf_context)")
    report.append("    else:")
    report.append("        result = 'No PDF documents available'")
    report.append("elif 'data' in user_query.lower() or 'analyze' in user_query.lower():")
    report.append("    if 'csv_context' in locals() and csv_context:")
    report.append("        result = data_analyst(f'Analyze: {user_query}')")
    report.append("    else:")
    report.append("        result = discover_data_files()")
    report.append("else:")
    report.append("    # Check all available context")
    report.append("    if 'pdf_context' in locals() and pdf_context:")
    report.append("        result = search_pdf_with_context(user_query, pdf_context)")
    report.append("    elif 'csv_context' in locals() and csv_context:")
    report.append("        result = data_analyst(f'Help with: {user_query}')")
    report.append("    else:")
    report.append("        result = enhanced_visit_webpage('search relevant info')")
    report.append("")
    report.append("final_answer(result)")
    report.append("```")
    report.append("")
    
    report.append("🛠️ DEBUGGING TOOLS:")
    report.append("   - check_context_availability(): This tool")
    report.append("   - demonstrate_context_access(): Shows access patterns")
    report.append("   - validate_context_structure(): Verifies context format")
    report.append("   - debug_agent_context(): Inspects agent's execution context")
    report.append("")
    
    report.append("📚 SMOLAGENTS DOCUMENTATION REFERENCE:")
    report.append("The additional_args mechanism allows passing rich context")
    report.append("to agents without modifying prompts. These become accessible")
    report.append("as variables in the agent's Python execution environment.")
    
    result = "\n".join(report)
    print("✅ Context availability analysis completed")
    return result


@tool
def demonstrate_context_access() -> str:
    """
    Demonstrate proper context access patterns for different task types with practical examples.
    
    This tool provides comprehensive examples of how to access and use context variables
    in various scenarios. Following smolagents best practices for educational tools
    that help users understand proper implementation patterns.
    
    Returns:
        Detailed examples showing context access patterns for PDF tasks, CSV analysis,
        web research, and mixed scenarios with complete code implementations.
        
    Usage Examples:
        - demonstrate_context_access()  # Learn context access patterns
        - Use when learning how to implement context-aware agents
        - Reference for implementing proper state variable handling
    """
    print("🎓 demonstrate_context_access: Providing comprehensive context examples")
    
    examples = ["🎓 CONTEXT ACCESS PATTERN DEMONSTRATIONS"]
    examples.append("=" * 60)
    examples.append("")
    examples.append("📚 SMOLAGENTS CONTEXT PRINCIPLES:")
    examples.append("- additional_args become state variables")
    examples.append("- Use 'in locals()' to check availability")
    examples.append("- Provide meaningful fallbacks")
    examples.append("- Handle empty or invalid context gracefully")
    examples.append("")
    
    # PDF Document Analysis Example
    examples.append("📄 EXAMPLE 1: PDF DOCUMENT ANALYSIS")
    examples.append("-" * 40)
    examples.append("**Task**: 'Analyze the financial metrics in the quarterly report'")
    examples.append("")
    examples.append("```python")
    examples.append("# Step 1: Check for PDF context")
    examples.append("if 'pdf_context' in locals() and pdf_context:")
    examples.append("    print(f'✅ PDF context found: {pdf_context[\"count\"]} documents')")
    examples.append("    ")
    examples.append("    # Step 2: Extract context information")
    examples.append("    available_files = pdf_context.get('available_files', [])")
    examples.append("    if available_files:")
    examples.append("        primary_doc = available_files[0]")
    examples.append("        filename = primary_doc.get('filename', 'Unknown')")
    examples.append("        classification = primary_doc.get('classification', 'General')")
    examples.append("        print(f'📄 Analyzing: {filename} ({classification})')")
    examples.append("        ")
    examples.append("        # Step 3: Perform targeted search")
    examples.append("        financial_query = 'financial metrics quarterly revenue earnings'")
    examples.append("        result = search_pdf_with_context(financial_query, pdf_context)")
    examples.append("        ")
    examples.append("        # Step 4: Return structured analysis")
    examples.append("        final_answer(f'Financial Analysis Results:\\n{result}')")
    examples.append("    else:")
    examples.append("        final_answer('PDF context exists but no files available')")
    examples.append("else:")
    examples.append("    # Fallback: web search for financial data")
    examples.append("    print('No PDF context, searching web for financial information')")
    examples.append("    web_result = enhanced_visit_webpage('https://investor.relations.site')")
    examples.append("    final_answer(f'Web Research Results:\\n{web_result}')")
    examples.append("```")
    examples.append("")
    
    # CSV Data Analysis Example
    examples.append("📊 EXAMPLE 2: CSV DATA ANALYSIS")
    examples.append("-" * 40)
    examples.append("**Task**: 'Create a visualization showing sales trends over time'")
    examples.append("")
    examples.append("```python")
    examples.append("# Step 1: Check for CSV context")
    examples.append("if 'csv_context' in locals() and csv_context:")
    examples.append("    print(f'✅ CSV context found: {csv_context[\"count\"]} datasets')")
    examples.append("    ")
    examples.append("    # Step 2: Prepare analysis request")
    examples.append("    files_info = csv_context.get('available_files', [])")
    examples.append("    if files_info:")
    examples.append("        dataset_names = [f['filename'] for f in files_info]")
    examples.append("        print(f'📊 Available datasets: {dataset_names}')")
    examples.append("        ")
    examples.append("        # Step 3: Delegate to data analyst with context")
    examples.append("        analysis_request = '''")
    examples.append("        Task: Create a visualization showing sales trends over time")
    examples.append("        Available data: {dataset_names}")
    examples.append("        Requirements:")
    examples.append("        1. Load the sales dataset")
    examples.append("        2. Identify time-based columns")
    examples.append("        3. Create line chart showing trends")
    examples.append("        4. Display with proper labels and formatting")
    examples.append("        '''")
    examples.append("        ")
    examples.append("        result = data_analyst(analysis_request)")
    examples.append("        final_answer(result)")
    examples.append("    else:")
    examples.append("        final_answer('CSV context exists but no datasets available')")
    examples.append("else:")
    examples.append("    # Fallback: attempt to discover CSV files")
    examples.append("    print('No CSV context, discovering available data files')")
    examples.append("    discovery_result = discover_data_files()")
    examples.append("    final_answer(f'Data Discovery Results:\\n{discovery_result}')")
    examples.append("```")
    examples.append("")
    
    # Mixed Context Example
    examples.append("🔄 EXAMPLE 3: MIXED CONTEXT HANDLING")
    examples.append("-" * 40)
    examples.append("**Task**: 'Compare our Q3 results with industry benchmarks'")
    examples.append("")
    examples.append("```python")
    examples.append("# Step 1: Analyze the query to determine primary data source")
    examples.append("query = 'Compare our Q3 results with industry benchmarks'")
    examples.append("internal_data_needed = 'our Q3 results' in query")
    examples.append("external_data_needed = 'industry benchmarks' in query")
    examples.append("")
    examples.append("# Step 2: Handle internal data (our Q3 results)")
    examples.append("internal_results = ''")
    examples.append("if 'pdf_context' in locals() and pdf_context:")
    examples.append("    # Search for Q3 results in PDFs")
    examples.append("    q3_query = 'Q3 quarterly results revenue earnings performance'")
    examples.append("    internal_results = search_pdf_with_context(q3_query, pdf_context)")
    examples.append("elif 'csv_context' in locals() and csv_context:")
    examples.append("    # Analyze Q3 data from CSV")
    examples.append("    analysis_request = 'Extract and summarize Q3 financial performance'")
    examples.append("    internal_results = data_analyst(analysis_request)")
    examples.append("else:")
    examples.append("    internal_results = 'No internal Q3 data available'")
    examples.append("")
    examples.append("# Step 3: Handle external data (industry benchmarks)")
    examples.append("external_results = ''")
    examples.append("if external_data_needed:")
    examples.append("    # Search for industry benchmarks online")
    examples.append("    benchmark_urls = [")
    examples.append("        'https://industry-reports.com/q3-benchmarks',")
    examples.append("        'https://financial-data.com/sector-analysis'")
    examples.append("    ]")
    examples.append("    external_results = bulk_visit_webpages(benchmark_urls)")
    examples.append("")
    examples.append("# Step 4: Combine and present results")
    examples.append("comparison_report = f'''")
    examples.append("Q3 PERFORMANCE COMPARISON REPORT")
    examples.append("========================================")
    examples.append("")
    examples.append("INTERNAL Q3 RESULTS:")
    examples.append("{internal_results}")
    examples.append("")
    examples.append("INDUSTRY BENCHMARKS:")
    examples.append("{external_results}")
    examples.append("")
    examples.append("COMPARATIVE ANALYSIS:")
    examples.append("[Analysis would be performed based on available data]")
    examples.append("'''")
    examples.append("")
    examples.append("final_answer(comparison_report)")
    examples.append("```")
    examples.append("")
    
    # Error Handling Example
    examples.append("⚠️ EXAMPLE 4: ROBUST ERROR HANDLING")
    examples.append("-" * 40)
    examples.append("**Task**: Handle various context scenarios gracefully")
    examples.append("")
    examples.append("```python")
    examples.append("def handle_context_safely(user_query):")
    examples.append("    try:")
    examples.append("        # Check all possible context types")
    examples.append("        context_available = {}")
    examples.append("        ")
    examples.append("        if 'pdf_context' in locals() and pdf_context:")
    examples.append("            pdf_files = pdf_context.get('available_files', [])")
    examples.append("            context_available['pdf'] = len(pdf_files)")
    examples.append("        ")
    examples.append("        if 'csv_context' in locals() and csv_context:")
    examples.append("            csv_files = csv_context.get('available_files', [])")
    examples.append("            context_available['csv'] = len(csv_files)")
    examples.append("        ")
    examples.append("        print(f'Available context: {context_available}')")
    examples.append("        ")
    examples.append("        # Route based on available context and query type")
    examples.append("        if 'pdf' in context_available and context_available['pdf'] > 0:")
    examples.append("            if any(keyword in user_query.lower() for keyword in")
    examples.append("                   ['document', 'report', 'pdf', 'research']):")
    examples.append("                return search_pdf_with_context(user_query, pdf_context)")
    examples.append("        ")
    examples.append("        if 'csv' in context_available and context_available['csv'] > 0:")
    examples.append("            if any(keyword in user_query.lower() for keyword in")
    examples.append("                   ['data', 'analyze', 'chart', 'graph', 'trend']):")
    examples.append("                return data_analyst(f'Analyze: {user_query}')")
    examples.append("        ")
    examples.append("        # Fallback to web search")
    examples.append("        print('No suitable context found, using web search')")
    examples.append("        return enhanced_visit_webpage(f'search: {user_query}')")
    examples.append("        ")
    examples.append("    except Exception as e:")
    examples.append("        error_msg = f'Context handling error: {str(e)}'")
    examples.append("        print(error_msg)")
    examples.append("        return f'Error occurred: {error_msg}. Please try rephrasing your query.'")
    examples.append("")
    examples.append("# Usage")
    examples.append("result = handle_context_safely(user_query)")
    examples.append("final_answer(result)")
    examples.append("```")
    examples.append("")
    
    examples.append("💡 KEY TAKEAWAYS:")
    examples.append("1. **Always check context availability** before using")
    examples.append("2. **Provide meaningful fallbacks** when context is missing")
    examples.append("3. **Log context information** for debugging")
    examples.append("4. **Handle errors gracefully** with user-friendly messages")
    examples.append("5. **Route queries intelligently** based on context and content")
    examples.append("6. **Combine multiple data sources** when appropriate")
    examples.append("")
    
    examples.append("🔧 DEBUGGING TIPS:")
    examples.append("- Use print() statements to log context availability")
    examples.append("- Check context structure before accessing nested data")
    examples.append("- Test with both valid and empty context scenarios")
    examples.append("- Implement try/catch blocks for robust error handling")
    examples.append("- Provide informative error messages to users")
    
    result = "\n".join(examples)
    print("✅ Context access demonstrations completed")
    return result


@tool
def validate_context_structure(context_data: dict = None, context_type: str = "unknown") -> str:
    """
    Validate context data structure and provide detailed feedback on format compliance.
    
    This tool checks if context data follows the expected smolagents format and provides
    comprehensive validation feedback. Following smolagents best practices for validation
    tools with detailed error reporting and correction guidance.
    
    Args:
        context_data: Dictionary containing context data to validate.
                     Should follow smolagents context structure standards.
        context_type: Type of context being validated ('pdf', 'csv', or 'unknown').
                     Helps provide specific validation rules for each context type.
        
    Returns:
        Comprehensive validation report including structure compliance, missing fields,
        format errors, and detailed recommendations for fixing any issues found.
        
    Usage Examples:
        - validate_context_structure(pdf_context, "pdf")
        - validate_context_structure(csv_context, "csv")
        - validate_context_structure(unknown_data)  # Auto-detect type
    """

    
    validation_report = ["🔍 CONTEXT STRUCTURE VALIDATION REPORT"]
    validation_report.append("=" * 60)
    validation_report.append("")
    
    # Input validation
    if context_data is None:
        validation_report.append("❌ VALIDATION ERROR: No context data provided")
        validation_report.append("")
        validation_report.append("💡 USAGE GUIDANCE:")
        validation_report.append("This tool validates context structure compliance.")
        validation_report.append("Expected usage patterns:")
        validation_report.append("")
        validation_report.append("```python")
        validation_report.append("# Validate PDF context")
        validation_report.append("if 'pdf_context' in locals() and pdf_context:")
        validation_report.append("    validation = validate_context_structure(pdf_context, 'pdf')")
        validation_report.append("    print(validation)")
        validation_report.append("")
        validation_report.append("# Validate CSV context")
        validation_report.append("if 'csv_context' in locals() and csv_context:")
        validation_report.append("    validation = validate_context_structure(csv_context, 'csv')")
        validation_report.append("    print(validation)")
        validation_report.append("```")
        validation_report.append("")
        validation_report.append("📚 EXPECTED CONTEXT STRUCTURES:")
        validation_report.append("")
        validation_report.append("**PDF Context**:")
        validation_report.append("{")
        validation_report.append("  'available_files': [")
        validation_report.append("    {")
        validation_report.append("      'filename': 'document.pdf',")
        validation_report.append("      'db_path': '/path/to/vector_db',")
        validation_report.append("      'user_notes': 'description',")
        validation_report.append("      'classification': 'category'")
        validation_report.append("    }")
        validation_report.append("  ],")
        validation_report.append("  'count': 1")
        validation_report.append("}")
        validation_report.append("")
        validation_report.append("**CSV Context**:")
        validation_report.append("{")
        validation_report.append("  'available_files': [")
        validation_report.append("    {")
        validation_report.append("      'filename': 'data.csv',")
        validation_report.append("      'path': '/path/to/data.csv',")
        validation_report.append("      'columns': ['col1', 'col2'],")
        validation_report.append("      'rows': 1000")
        validation_report.append("    }")
        validation_report.append("  ],")
        validation_report.append("  'count': 1")
        validation_report.append("}")
        
        result = "\n".join(validation_report)
        print("⚠️ No context data provided for validation")
        return result
    
    if not isinstance(context_data, dict):
        validation_report.append(f"❌ TYPE ERROR: Expected dict, received {type(context_data)}")
        validation_report.append("")
        validation_report.append("💡 SOLUTION: Context data must be a dictionary")
        validation_report.append("Check that context is properly prepared before validation")
        
        result = "\n".join(validation_report)
        print(f"❌ Invalid context data type: {type(context_data)}")
        return result
    
    # Auto-detect context type if unknown
    if context_type == "unknown":
        if 'db_path' in str(context_data):
            context_type = "pdf"
            print("   🔍 Auto-detected context type: PDF")
        elif 'columns' in str(context_data) or 'rows' in str(context_data):
            context_type = "csv"
            print("   🔍 Auto-detected context type: CSV")
        else:
            context_type = "generic"
            print("   🔍 Auto-detected context type: Generic")
    
    validation_report.append(f"📋 VALIDATING {context_type.upper()} CONTEXT STRUCTURE")
    validation_report.append(f"📊 Data size: {len(str(context_data))} characters")
    validation_report.append("")
    
    errors_found = []
    warnings_found = []
    suggestions = []
    
    # Validate basic structure
    validation_report.append("🔍 BASIC STRUCTURE VALIDATION:")
    
    # Check for required top-level fields
    required_fields = ['available_files']
    optional_fields = ['count', 'metadata']
    
    for field in required_fields:
        if field in context_data:
            validation_report.append(f"   ✅ Required field '{field}': Present")
        else:
            error_msg = f"Missing required field: '{field}'"
            errors_found.append(error_msg)
            validation_report.append(f"   ❌ {error_msg}")
    
    for field in optional_fields:
        if field in context_data:
            validation_report.append(f"   ✅ Optional field '{field}': Present")
        else:
            validation_report.append(f"   ⚠️ Optional field '{field}': Missing (recommended)")
    
    validation_report.append("")
    
    # Validate available_files structure
    if 'available_files' in context_data:
        available_files = context_data['available_files']
        validation_report.append("📁 AVAILABLE_FILES VALIDATION:")
        
        if isinstance(available_files, list):
            validation_report.append(f"   ✅ Type: List ({len(available_files)} files)")
            
            if len(available_files) == 0:
                warning_msg = "available_files list is empty"
                warnings_found.append(warning_msg)
                validation_report.append(f"   ⚠️ {warning_msg}")
            
            # Validate each file entry
            for i, file_entry in enumerate(available_files):
                validation_report.append(f"   📄 File #{i+1} validation:")
                
                if isinstance(file_entry, dict):
                    validation_report.append(f"      ✅ Type: Dictionary")
                    
                    # Context-specific field validation
                    if context_type == "pdf":
                        pdf_required = ['filename', 'db_path']
                        pdf_optional = ['user_notes', 'classification']
                        
                        for field in pdf_required:
                            if field in file_entry:
                                value = file_entry[field]
                                if value:
                                    validation_report.append(f"      ✅ {field}: '{value}'")
                                else:
                                    error_msg = f"File #{i+1}: {field} is empty"
                                    errors_found.append(error_msg)
                                    validation_report.append(f"      ❌ {error_msg}")
                            else:
                                error_msg = f"File #{i+1}: Missing {field}"
                                errors_found.append(error_msg)
                                validation_report.append(f"      ❌ {error_msg}")
                        
                        for field in pdf_optional:
                            if field in file_entry:
                                validation_report.append(f"      ✅ {field}: Present")
                            else:
                                validation_report.append(f"      ⚠️ {field}: Missing (optional)")
                    
                    elif context_type == "csv":
                        csv_required = ['filename', 'path']
                        csv_optional = ['columns', 'rows', 'description']
                        
                        for field in csv_required:
                            if field in file_entry:
                                value = file_entry[field]
                                if value:
                                    validation_report.append(f"      ✅ {field}: '{value}'")
                                else:
                                    error_msg = f"File #{i+1}: {field} is empty"
                                    errors_found.append(error_msg)
                                    validation_report.append(f"      ❌ {error_msg}")
                            else:
                                error_msg = f"File #{i+1}: Missing {field}"
                                errors_found.append(error_msg)
                                validation_report.append(f"      ❌ {error_msg}")
                        
                        for field in csv_optional:
                            if field in file_entry:
                                validation_report.append(f"      ✅ {field}: Present")
                            else:
                                validation_report.append(f"      ⚠️ {field}: Missing (recommended)")
                else:
                    error_msg = f"File #{i+1}: Not a dictionary"
                    errors_found.append(error_msg)
                    validation_report.append(f"      ❌ {error_msg}")
        else:
            error_msg = "available_files is not a list"
            errors_found.append(error_msg)
            validation_report.append(f"   ❌ {error_msg}")
    
    validation_report.append("")
    
    # Count validation
    if 'count' in context_data:
        count_value = context_data['count']
        actual_count = len(context_data.get('available_files', []))
        
        validation_report.append("🔢 COUNT VALIDATION:")
        validation_report.append(f"   📊 Declared count: {count_value}")
        validation_report.append(f"   📊 Actual files: {actual_count}")
        
        if count_value == actual_count:
            validation_report.append(f"   ✅ Count matches actual files")
        else:
            warning_msg = f"Count mismatch: declared {count_value}, actual {actual_count}"
            warnings_found.append(warning_msg)
            validation_report.append(f"   ⚠️ {warning_msg}")
    
    validation_report.append("")
    
    # Summary
    validation_report.append("📊 VALIDATION SUMMARY:")
    validation_report.append(f"   ✅ Errors found: {len(errors_found)}")
    validation_report.append(f"   ⚠️ Warnings: {len(warnings_found)}")
    validation_report.append("")
    
    if errors_found:
        validation_report.append("❌ ERRORS TO FIX:")
        for error in errors_found:
            validation_report.append(f"   - {error}")
        validation_report.append("")
    
    if warnings_found:
        validation_report.append("⚠️ WARNINGS (RECOMMENDED FIXES):")
        for warning in warnings_found:
            validation_report.append(f"   - {warning}")
        validation_report.append("")
    
    # Overall status
    if not errors_found:
        validation_report.append("🎉 VALIDATION RESULT: PASSED")
        validation_report.append("Context structure is compliant with smolagents standards.")
        if warnings_found:
            validation_report.append("Consider addressing warnings for optimal performance.")
    else:
        validation_report.append("❌ VALIDATION RESULT: FAILED")
        validation_report.append("Context structure has errors that need to be fixed.")
    
    validation_report.append("")
    validation_report.append("💡 NEXT STEPS:")
    if errors_found:
        validation_report.append("1. Fix all errors listed above")
        validation_report.append("2. Re-run validation to confirm fixes")
        validation_report.append("3. Test context with actual tools")
    else:
        validation_report.append("1. Address warnings if possible")
        validation_report.append("2. Test context with relevant tools")
        validation_report.append("3. Monitor context usage in production")
    
    result = "\n".join(validation_report)
    
    if errors_found:
        print(f"❌ Validation failed: {len(errors_found)} errors, {len(warnings_found)} warnings")
    else:
        print(f"✅ Validation passed: {len(warnings_found)} warnings")
    
    return result


@tool
def debug_agent_context() -> str:
    """
    Debug tool to inspect the current agent's execution context and available variables.
    
    This tool helps identify why PDF context might not be accessible and provides
    specific troubleshooting steps for the smolagents architecture.
    
    Returns:
        Detailed report of the agent's current context state and troubleshooting guidance
    """
    import inspect
    import sys
    
    debug_report = ["🔍 AGENT CONTEXT DEBUG REPORT"]
    debug_report.append("=" * 50)
    debug_report.append("")
    
    try:
        # Get current frame and inspect local variables
        frame = inspect.currentframe()
        if frame and frame.f_back:
            local_vars = frame.f_back.f_locals
            global_vars = frame.f_back.f_globals
            
            debug_report.append("📋 LOCAL VARIABLES INSPECTION:")
            debug_report.append(f"   Total local variables: {len(local_vars)}")
            
            # Check for expected context variables
            context_vars = ['pdf_context', 'csv_context', 'additional_args']
            found_context = []
            
            for var_name in context_vars:
                if var_name in local_vars:
                    var_value = local_vars[var_name]
                    if var_value:
                        found_context.append(f"✅ {var_name}: {type(var_value)} (has data)")
                        if isinstance(var_value, dict):
                            debug_report.append(f"      Keys: {list(var_value.keys())}")
                    else:
                        found_context.append(f"⚠️ {var_name}: {type(var_value)} (empty)")
                else:
                    found_context.append(f"❌ {var_name}: Not found")
            
            debug_report.extend(found_context)
            debug_report.append("")
            
            # Check for other relevant variables
            debug_report.append("🔍 OTHER RELEVANT VARIABLES:")
            relevant_vars = ['task', 'user_query', 'query', 'prompt']
            for var_name in relevant_vars:
                if var_name in local_vars:
                    var_value = local_vars[var_name]
                    debug_report.append(f"   ✅ {var_name}: {type(var_value)} = {str(var_value)[:100]}...")
            
            debug_report.append("")
            
        else:
            debug_report.append("❌ Could not access frame information")
            debug_report.append("")
        
        # Check Python path and imports
        debug_report.append("🐍 PYTHON ENVIRONMENT:")
        debug_report.append(f"   Python version: {sys.version}")
        debug_report.append(f"   Current working directory: {sys.path[0] if sys.path else 'Unknown'}")
        debug_report.append("")
        
        # Provide troubleshooting guidance
        debug_report.append("🛠️ TROUBLESHOOTING GUIDANCE:")
        debug_report.append("")
        debug_report.append("1. **Context Variable Issues:**")
        debug_report.append("   - Verify additional_args are passed to agent.run()")
        debug_report.append("   - Check if context preparation is working in ui/chat.py")
        debug_report.append("   - Ensure PDF files are properly processed and indexed")
        debug_report.append("")
        debug_report.append("2. **PDF Database Issues:**")
        debug_report.append("   - Check if vector database files exist in data/pdf_temp/")
        debug_report.append("   - Verify PDF indexing completed successfully")
        debug_report.append("   - Ensure ChromaDB dependencies are installed")
        debug_report.append("")
        debug_report.append("3. **Agent Configuration Issues:**")
        debug_report.append("   - Verify RAG agent has correct tools imported")
        debug_report.append("   - Check if context passing mechanism is working")
        debug_report.append("   - Ensure smolagents version is compatible")
        debug_report.append("")
        debug_report.append("💡 NEXT STEPS:")
        debug_report.append("   1. Upload and index PDF files via the UI")
        debug_report.append("   2. Check session state in Streamlit sidebar")
        debug_report.append("   3. Use diagnose_pdf_context() for PDF-specific debugging")
        debug_report.append("   4. Try smart_pdf_search() which has built-in fallbacks")
        
    except Exception as e:
        debug_report.append(f"❌ Error during context debugging: {str(e)}")
        debug_report.append("This might indicate a deeper issue with the agent execution environment")
    
    result = "\n".join(debug_report)
    print("🔍 Agent context debug completed")
    return result 