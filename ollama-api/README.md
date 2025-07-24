# Ollama LLM Integration Service - AI Chat and Vision Analysis

Advanced LLM integration service providing both text generation and image analysis capabilities through Ollama. Offers both REST API and Discord bot interfaces with dynamic model selection.

## Features

- **Text Generation**: Chat with various LLM models (Llama2, Mistral, CodeLlama, etc.)
- **Image Analysis**: Vision models for image description and analysis
- **Model Management**: Switch between pre-installed models (no auto-installation for security)
- **Dual Interfaces**: RESTful API and Discord bot integration
- **Multiple Input Methods**: URL, file upload, and direct text prompts
- **Streaming Support**: Fast response delivery (API ready, Discord uses chunking)
- **Error Resilience**: Graceful handling of Ollama connectivity issues

## Quick Start

### Prerequisites

1. **Install Ollama**:
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Pull some models
ollama pull llama2
ollama pull llava  # For image analysis
ollama pull mistral
```

2. **Install Python dependencies**:
```bash
pip install flask python-dotenv ollama discord.py requests
```

### Configuration

1. Copy environment template:
```bash
cp .env.sample .env
```

2. Edit `.env` file:
```bash
# Service Settings
PORT=7782
PRIVATE=false

# API Configuration (required for emoji lookup)
API_HOST=localhost
API_PORT=8080
API_TIMEOUT=5.0

# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
DEFAULT_TEXT_MODEL=llama2
DEFAULT_VISION_MODEL=llava

# Discord Bot (optional)
DISCORD_TOKEN=your_bot_token
DISCORD_GUILD=your_server_name
DISCORD_CHANNEL=general,ai-chat
```

### Running the Services

#### REST API Only
```bash
# Direct execution
python3 REST.py

# Using startup script
./llama.sh

# As systemd service
sudo systemctl start llama-api
sudo systemctl enable llama-api
```

#### Discord Bot
```bash
# Direct execution
python3 discord-bot.py

# Using startup script
./discord.sh

# As systemd service
sudo systemctl start llama
sudo systemctl enable llama
```

## REST API Usage

### Health Check
```bash
GET /health
```

Returns service status, available models, and Ollama connectivity.

### List Available Models
```bash
GET /models
```

### Text Generation
```bash
POST /text
Content-Type: application/json

{
  "prompt": "Explain quantum computing in simple terms",
  "model": "llama2"  # optional, uses default if not specified
}
```

### Image Analysis
```bash
POST /image
Content-Type: multipart/form-data

# Form fields:
# - uploadedfile: image file
# - prompt: custom analysis prompt (optional)
# - model: vision model to use (optional)
```

### Legacy Image Analysis (URL)
```bash
GET /?url=https://example.com/image.jpg&prompt=Describe this image
```

## API Response Format

### Text Generation Response
```json
{
  "llm": {
    "response": "Quantum computing is a type of computing that uses quantum bits...",
    "model_used": "llama2",
    "prompt_length": 45,
    "response_length": 234,
    "processing_time": 2.456,
    "type": "text",
    "status": "success"
  }
}
```

### Image Analysis Response
```json
{
  "llm": {
    "response": "This image shows a beautiful sunset over a mountain range...",
    "model_used": "llava",
    "prompt": "Describe this image in detail.",
    "response_length": 156,
    "processing_time": 3.234,
    "type": "vision",
    "status": "success"
  }
}
```

## Discord Bot Usage

### Text Commands
- Just type a message to chat with the current model
- `/ask <question>` - Ask a specific question
- `/chat <message>` - Alternative chat command

### Model Management
- `/list` - Show all available models with current selection
- `/use <model_name>` or `/use <number>` - Switch models
- `/model` - Show current active model

### Image Analysis
- Upload any image to analyze it automatically
- Add text with your image for custom analysis prompts
- Supports: JPG, PNG, GIF, BMP, WEBP

### Help and Info
- `/help` - Show detailed command help

### Example Discord Usage
```
User: Hello, how are you?
Bot: Hello! I'm doing well, thank you for asking. I'm an AI assistant powered by Llama2...

User: /use mistral
Bot: ✅ Model set to: mistral

User: /ask What is the capital of France?
Bot: The capital of France is Paris...

User: [uploads image of a cat] What breed is this?
Bot: This appears to be a Maine Coon cat, characterized by...
```

## Available Models

### Text Models
- **llama2**: General-purpose conversational AI
- **llama2:13b**: Larger version with better performance
- **mistral**: Fast and efficient instruction-following
- **mixtral**: Mixture of experts model
- **codellama**: Specialized for code generation
- **dolphin-mistral**: Uncensored conversational model
- **phi**: Microsoft's small but capable model

### Vision Models
- **llava**: LLaVA (Large Language and Vision Assistant)
- **llava:13b**: Larger version for better image understanding
- **bakllava**: Alternative vision model

### Code Models
- **codellama**: General code assistant
- **codellama:13b**: Enhanced code generation
- **deepseek-coder:6.7b**: Specialized coding model

## Performance and Limitations

### Performance Metrics
- **Text Generation**: 2-10 seconds typical (depends on model size and prompt)
- **Image Analysis**: 3-15 seconds typical (vision models are slower)
- **Memory Usage**: Varies by model (2GB-16GB+ VRAM recommended)
- **Concurrent Requests**: Limited by Ollama server capacity

### File Size Limits
- **Images**: 8MB maximum
- **Text Prompts**: 10,000 characters maximum
- **Response Length**: 4,000 characters (truncated if longer)

### Model Requirements
- **Small models** (7B params): 8GB+ VRAM recommended
- **Medium models** (13B params): 16GB+ VRAM recommended
- **Large models** (30B+ params): 24GB+ VRAM recommended
- **CPU-only**: Possible but much slower

## Error Handling

The service provides comprehensive error handling:

```json
{
  "error": "Text generation failed: Model 'llama2' not found",
  "status": "error"
}
```

Common error scenarios:
- Ollama server not running
- Model not available/not pulled
- Image too large or invalid format
- Network connectivity issues
- GPU memory exhaustion

## Integration with Animal Farm

This service integrates with the Animal Farm distributed AI ecosystem:

- **Multi-Modal AI**: Combines text and vision capabilities
- **Voting Ensemble**: LLM responses can inform decision algorithms
- **Discord Ecosystem**: Unified bot experience across all AI services
- **API Consistency**: Follows established patterns for easy integration

## Development and Customization

### Model Requirements
**Required Models (must be pre-installed):**
- `llava` - Essential for image analysis capability
- `llama2` - Default text generation model

**Additional Models (optional but recommended):**
```bash
# Install required models
ollama pull llava
ollama pull llama2

# Optional additional models
ollama pull mistral
ollama pull codellama
```

**Note**: The service only uses pre-installed models for security and consistency. No automatic model installation occurs.

### Custom Prompts for Images
```python
# REST API
{
  "prompt": "Analyze the artistic style and composition of this image",
  "model": "llava"
}

# Discord
# Upload image with message: "What art movement does this represent?"
```

### Extending the API
```python
# Add new endpoint for streaming responses
@app.route('/stream', methods=['POST'])
def stream_text():
    # Implementation for streaming responses
    pass
```

## Troubleshooting

### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# Start Ollama
ollama serve

# Check available models
ollama list
```

### Model Not Found
```bash
# Pull missing model
ollama pull llama2
ollama pull llava

# Verify model is available
ollama list
```

### Discord Bot Issues
1. **Bot not responding**: Check Discord token and permissions
2. **Channel restrictions**: Verify DISCORD_CHANNEL settings
3. **Model switching fails**: Ensure Ollama connectivity

### Performance Issues
1. **Slow responses**: Check GPU availability and memory
2. **Memory errors**: Use smaller models or increase VRAM
3. **Connection timeouts**: Increase timeout values in code

### Service Startup Issues
```bash
# Check service status
systemctl status llama-api
systemctl status llama

# View logs
journalctl -u llama-api -f
journalctl -u llama -f

# Test direct execution
python3 REST.py
python3 discord-bot.py
```

## Security Considerations

- **Model Access**: All users can switch models - consider restrictions in production
- **Input Validation**: Text prompts and images are validated for size
- **API Rate Limiting**: Consider implementing rate limiting for production use
- **Private Mode**: Use PRIVATE=true for localhost-only API access
- **Discord Permissions**: Restrict bot to specific channels as needed

## Dependencies

- `flask`: Web framework for REST API
- `python-dotenv`: Environment variable management
- `ollama`: Official Ollama Python client
- `discord.py`: Discord bot framework
- `requests`: HTTP client for image downloads

## License

Part of the Animal Farm distributed AI project.