# Personal AI Assistant ✨

A powerful, feature-rich AI assistant built with Streamlit and powered by Llama 3.3. This application combines conversational AI with practical tools, document analysis, web search, and persistent memory capabilities.

## 🌟 Features

### Core Capabilities
- **Conversational AI**: Powered by Llama 3.3 70B model via Groq API
- **Document Analysis**: Upload and analyze PDFs, text files, and more
- **Web Search**: Real-time Google search integration
- **Persistent Memory**: Remember conversations across sessions using Mem0
- **Multi-tool Integration**: Calculator, unit converter, text analyzer, and more

### Built-in Tools
- 🧮 **Calculator**: Mathematical expressions and calculations
- 📝 **Text Analyzer**: Word count, character count, paragraph analysis
- 💱 **Currency Converter**: Convert between major currencies
- 📏 **Unit Converter**: Length, weight, temperature, volume conversions
- ⏰ **Time Checker**: Current date and time
- 🔍 **Web Search**: Google Custom Search integration
- 📊 **Text Summarizer**: Summarize long text into key points
- 📄 **Document Search**: Query uploaded documents

### UI Features
- **Modern Design**: Gradient backgrounds with glassmorphism effects
- **Responsive Layout**: Works on desktop and mobile
- **Interactive Elements**: Hover effects, animations, and transitions
- **Dark Sidebar**: High contrast sidebar with status indicators
- **Sample Questions**: Quick-start buttons for common queries

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- API keys for:
  - Groq API (required)
  - Google Custom Search API (optional)
  - Mem0 API (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd personal-ai-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   GOOGLE_CX=your_google_custom_search_engine_id
   MEM0_API_KEY=your_mem0_api_key_here
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser**
   Navigate to `http://localhost:8501`

## 📋 Requirements

Create a `requirements.txt` file with these dependencies:

```txt
streamlit>=1.28.0
python-dotenv>=1.0.0
llama-index>=0.10.0
llama-index-llms-groq>=0.1.0
llama-index-embeddings-huggingface>=0.2.0
llama-index-memory-mem0>=0.1.0
requests>=2.31.0
pathlib>=1.0.0
tempfile
shutil
```

## 🔧 Configuration

### API Keys Setup

#### Groq API (Required)
1. Visit [Groq Console](https://console.groq.com/)
2. Create an account and generate an API key
3. Add to `.env`: `GROQ_API_KEY=your_key_here`

#### Google Custom Search (Optional)
1. Visit [Google Cloud Console](https://console.cloud.google.com/)
2. Enable Custom Search API
3. Create credentials and get API key
4. Set up Custom Search Engine at [Google CSE](https://cse.google.com/)
5. Add to `.env`:
   ```env
   GOOGLE_API_KEY=your_api_key
   GOOGLE_CX=your_search_engine_id
   ```

#### Mem0 API (Optional)
1. Visit [Mem0 Platform](https://app.mem0.ai/)
2. Create account and get API key
3. Add to `.env`: `MEM0_API_KEY=your_key_here`

### Model Configuration

The app uses these default settings (configurable in the code):
- **Model**: `llama-3.3-70b-versatile`
- **Embedding Model**: `BAAI/bge-small-en-v1.5`
- **Chunk Size**: 1000 tokens
- **Chunk Overlap**: 20 tokens
- **Output Tokens**: 1024

## 📖 Usage Guide

### Basic Chat
1. Type your question in the chat input
2. The AI will respond using its knowledge and available tools
3. Conversations are remembered if Mem0 is configured

### Document Analysis
1. Upload documents using the sidebar file uploader
2. Supported formats: PDF, TXT, MD, CSV, JSON
3. Ask questions about your documents
4. The AI will search through uploaded content

### Using Built-in Tools
The AI automatically selects appropriate tools based on your query:

- **Math**: "What's 15% of 250?" → Uses Calculator
- **Conversions**: "Convert 100 USD to EUR" → Uses Currency Converter
- **Text Analysis**: "Count words in this text" → Uses Text Analyzer
- **Web Search**: "Latest AI news" → Uses Web Search
- **Time**: "What time is it?" → Uses Time Checker

### Sample Questions
Use the sidebar sample questions for quick starts:
- Mathematical calculations
- Unit conversions
- Text analysis
- Web searches
- Document queries

## 🎨 Customization

### Styling
The app uses extensive CSS customization in the `st.markdown()` sections. Key design elements:
- **Gradient backgrounds**: Multiple color gradients
- **Glassmorphism effects**: Backdrop blur and transparency
- **Interactive elements**: Hover effects and animations
- **High contrast text**: Ensures readability

### Adding New Tools
To add custom tools:

1. **Create a function**:
   ```python
   def my_custom_tool(input_param: str) -> str:
       """Description of what the tool does"""
       # Your logic here
       return result
   ```

2. **Add to tools list**:
   ```python
   tools.append(
       FunctionTool.from_defaults(
           fn=my_custom_tool,
           name="CustomTool",
           description="Description for the AI agent"
       )
   )
   ```

### Modifying UI
- **Colors**: Update CSS gradient values
- **Layout**: Modify column layouts and spacing
- **Components**: Add new Streamlit components
- **Fonts**: Change font families in CSS

## 🛠️ Technical Details

### Architecture
- **Frontend**: Streamlit with custom CSS
- **LLM**: Llama 3.3 via Groq API
- **Vector Store**: LlamaIndex VectorStoreIndex
- **Embeddings**: HuggingFace BGE model
- **Memory**: Mem0 for conversation persistence
- **Agent**: ReActAgent with multiple tools

### File Structure
```
personal-ai-assistant/
├── app.py                 # Main application
├── .env                   # Environment variables
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── temp/                 # Temporary files (auto-created)
```

### Performance Optimization
- **Caching**: Streamlit caching for LLM and settings
- **Lazy Loading**: Components loaded on demand
- **Memory Management**: Temporary files cleaned up automatically

## 🔒 Security Notes

- **API Keys**: Never commit API keys to version control
- **File Upload**: Temporary files are automatically cleaned up
- **Expression Evaluation**: Calculator uses safe evaluation methods
- **Input Validation**: Tools validate inputs before processing

## 🐛 Troubleshooting

### Common Issues

**"No module named 'llama_index'"**
```bash
pip install llama-index
```

**"Invalid API key"**
- Check your `.env` file
- Verify API key is correct
- Ensure no extra spaces in the key

**"Document processing failed"**
- Check file format is supported
- Ensure file isn't corrupted
- Try with a smaller file first

**"Memory not working"**
- Verify MEM0_API_KEY is set
- Check Mem0 service status
- Try different user_id

### Debug Mode
Enable verbose logging by setting `verbose=True` in the ReActAgent initialization.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Groq**: For the Llama 3.3 API
- **LlamaIndex**: For the RAG framework
- **Streamlit**: For the web framework
- **HuggingFace**: For embeddings
- **Mem0**: For memory capabilities

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Search existing issues
3. Create a new issue with detailed description
4. Include error messages and environment details

---

**Built with ❤️ using Streamlit and Llama 3.3**
