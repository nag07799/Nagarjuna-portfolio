# AI Chatbot Integration Guide

## Overview
This portfolio now includes an AI-powered chatbot that can answer questions about Nagarjuna Pendekanti's professional background, experience, and skills using the Groq API with Llama 3.3 70B model.

## Features
- **Intelligent Q&A**: Answers questions about work experience, technical skills, projects, and education
- **Resume-Based Knowledge**: Trained on complete resume data with detailed context
- **Modern UI**: Beautiful gradient design matching the portfolio theme
- **Suggested Questions**: Quick-start prompts for common queries
- **Real-time Responses**: Fast inference using Groq's optimized infrastructure
- **Mobile Responsive**: Works seamlessly on desktop and mobile devices

## Setup Instructions

### 1. Environment Variables
The API key is stored in `.env` file:
```
REACT_APP_GROQ_API_KEY=your_api_key_here
```

**Important**: The `.env` file is already in `.gitignore` to keep your API key secure.

### 2. Running the Application
```bash
npm install
npm start
```

The chatbot will be accessible via the "Ask AI" button in the navigation bar.

## Architecture

### Files Created
1. **src/components/chatbot/Chatbot.js** - Main chatbot component with Groq integration
2. **src/data/resumeKnowledge.js** - Resume context and system prompts for the AI
3. **.env** - Environment variables for API configuration

### Files Modified
1. **src/App.js** - Added chatbot state management and integration
2. **src/components/navbar/Navbar.js** - Added "Ask AI" toggle button
3. **.gitignore** - Added .env to protect API keys

## Usage Examples

### Sample Questions You Can Ask:
1. "What's your experience with multi-agent systems?"
2. "Tell me about your fraud detection work"
3. "What are your key technical skills?"
4. "Describe your most impactful project"
5. "What's your educational background?"
6. "How did you reduce false positives in fraud detection?"
7. "What cloud technologies do you work with?"
8. "Tell me about your experience with LangChain and LangGraph"

## Technical Details

### Model Configuration
- **Model**: llama-3.3-70b-versatile
- **Temperature**: 0.7 (balanced creativity and accuracy)
- **Max Tokens**: 1024
- **Provider**: Groq (optimized LLM inference)

### Key Components
- **Context Management**: Full resume context provided in system prompt
- **Conversation History**: Multi-turn dialogue support
- **Error Handling**: Graceful fallbacks for API errors
- **Loading States**: Visual feedback during response generation

## Customization

### Updating Resume Information
Edit `src/data/resumeKnowledge.js` to update the chatbot's knowledge base.

### Styling
The chatbot uses Tailwind CSS classes matching your portfolio theme:
- Primary colors: `designColor` and gradient effects
- Background: `bodyColor` with dark theme
- Accent: Purple gradient (`#c026d3`)

### Model Selection
You can change the model in `Chatbot.js`:
```javascript
model: 'llama-3.3-70b-versatile', // Options: llama-3.3-70b-versatile, mixtral-8x7b-32768, etc.
```

## Security Notes
1. API key is stored in environment variables
2. `.env` is gitignored to prevent exposure
3. `dangerouslyAllowBrowser: true` is set for client-side usage (consider moving to backend for production)

## Future Enhancements
- [ ] Add conversation persistence (localStorage)
- [ ] Implement typing indicators
- [ ] Add voice input/output
- [ ] Create analytics for common questions
- [ ] Add feedback mechanism (thumbs up/down)
- [ ] Implement rate limiting
- [ ] Move API calls to backend for better security

## Troubleshooting

### Chatbot doesn't respond
- Check if `.env` file exists with valid API key
- Verify internet connection
- Check browser console for errors

### API Errors
- Ensure Groq API key is valid and has credits
- Check rate limits on your Groq account

### Styling Issues
- Ensure Tailwind CSS is properly configured
- Check that all required React Icons are installed

## Support
For issues or questions, refer to:
- [Groq API Documentation](https://console.groq.com/docs)
- [React Documentation](https://react.dev)
