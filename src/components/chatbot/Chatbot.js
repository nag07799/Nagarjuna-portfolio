import React, { useState, useRef, useEffect } from 'react';
import Groq from 'groq-sdk';
import { resumeContext, systemPrompt } from '../../data/resumeKnowledge';
import { FaPaperPlane, FaTimes, FaRobot, FaUser } from 'react-icons/fa';

const Chatbot = ({ isOpen, onClose }) => {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: "Hey! I'm Nagarjuna. Ask me anything about my resume, experience, skills, or projects and I'll answer just like I would in person."
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const chatContainerRef = useRef(null);

  // Initialize Groq client with API key
  const groq = new Groq({
    apiKey: process.env.REACT_APP_GROQ_API_KEY,
    dangerouslyAllowBrowser: true // Required for client-side usage
  });

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput('');

    // Add user message to chat
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsLoading(true);

    try {
      // Prepare messages for Groq API
      const chatMessages = [
        {
          role: 'system',
          content: systemPrompt + '\n\n' + resumeContext
        },
        ...messages.map(msg => ({
          role: msg.role,
          content: msg.content
        })),
        {
          role: 'user',
          content: userMessage
        }
      ];

      // Call Groq API
      const completion = await groq.chat.completions.create({
        messages: chatMessages,
        model: 'llama-3.3-70b-versatile', // Using Llama 3.3 70B model
        temperature: 0.7,
        max_tokens: 1024,
        top_p: 1,
        stream: false
      });

      const assistantMessage = completion.choices[0]?.message?.content ||
        "I apologize, but I couldn't generate a response. Please try again.";

      // Add assistant response to chat
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: assistantMessage
      }]);
    } catch (error) {
      console.error('Error calling Groq API:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: "I'm sorry, I encountered an error processing your request. Please try again."
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const suggestedQuestions = [
    "What's your experience with multi-agent systems?",
    "Tell me about your fraud detection work",
    "What are your key technical skills?",
    "Describe your most impactful project",
  ];

  const handleSuggestedQuestion = (question) => {
    setInput(question);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed bottom-4 right-4 w-96 h-[600px] bg-gradient-to-r from-bodyColor to-[#1a1a2e] border border-designColor rounded-lg shadow-2xl flex flex-col z-50">
      {/* Header */}
      <div className="bg-gradient-to-r from-designColor to-[#c026d3] p-4 rounded-t-lg flex justify-between items-center">
        <div className="flex items-center gap-2">
          <FaRobot className="text-white text-xl" />
          <div>
            <h3 className="text-white font-bold text-lg">Ask Nagarjuna</h3>
            <p className="text-gray-200 text-xs">Ask me about my resume</p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="text-white hover:text-gray-200 transition-colors"
        >
          <FaTimes className="text-xl" />
        </button>
      </div>

      {/* Messages Container */}
      <div
        ref={chatContainerRef}
        className="flex-1 overflow-y-auto p-4 space-y-4 bg-bodyColor"
      >
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex gap-2 ${
              message.role === 'user' ? 'justify-end' : 'justify-start'
            }`}
          >
            {message.role === 'assistant' && (
              <div className="w-8 h-8 rounded-full bg-designColor flex items-center justify-center flex-shrink-0">
                <FaRobot className="text-white text-sm" />
              </div>
            )}
            <div
              className={`max-w-[75%] rounded-lg p-3 ${
                message.role === 'user'
                  ? 'bg-designColor text-white'
                  : 'bg-[#1a1a2e] text-lightText border border-zinc-800'
              }`}
            >
              <p className="text-sm leading-relaxed whitespace-pre-wrap">
                {message.content}
              </p>
            </div>
            {message.role === 'user' && (
              <div className="w-8 h-8 rounded-full bg-gradient-to-r from-[#c026d3] to-designColor flex items-center justify-center flex-shrink-0">
                <FaUser className="text-white text-sm" />
              </div>
            )}
          </div>
        ))}

        {isLoading && (
          <div className="flex gap-2 justify-start">
            <div className="w-8 h-8 rounded-full bg-designColor flex items-center justify-center flex-shrink-0">
              <FaRobot className="text-white text-sm" />
            </div>
            <div className="bg-[#1a1a2e] text-lightText rounded-lg p-3 border border-zinc-800">
              <div className="flex gap-1">
                <span className="w-2 h-2 bg-designColor rounded-full animate-bounce"></span>
                <span className="w-2 h-2 bg-designColor rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></span>
                <span className="w-2 h-2 bg-designColor rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Suggested Questions */}
      {messages.length === 1 && (
        <div className="px-4 py-2 bg-bodyColor border-t border-zinc-800">
          <p className="text-xs text-gray-400 mb-2">Suggested questions:</p>
          <div className="grid grid-cols-1 gap-1">
            {suggestedQuestions.map((question, index) => (
              <button
                key={index}
                onClick={() => handleSuggestedQuestion(question)}
                className="text-left text-xs text-designColor hover:text-white bg-[#1a1a2e] hover:bg-designColor/20 px-2 py-1 rounded transition-colors"
              >
                {question}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="p-4 bg-bodyColor border-t border-zinc-800">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask me about my resume, experience, or skills..."
            disabled={isLoading}
            className="flex-1 bg-[#1a1a2e] text-lightText px-4 py-2 rounded-lg border border-zinc-800 focus:border-designColor focus:outline-none text-sm disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="bg-gradient-to-r from-designColor to-[#c026d3] text-white px-4 py-2 rounded-lg hover:shadow-lg hover:shadow-designColor/50 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <FaPaperPlane />
          </button>
        </div>
      </form>
    </div>
  );
};

export default Chatbot;
