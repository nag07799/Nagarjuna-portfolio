import React from 'react';
import { FaRobot, FaTimes } from 'react-icons/fa';

const ChatbotButton = ({ isOpen, onClick }) => {
  return (
    <button
      onClick={onClick}
      className={`fixed bottom-4 right-4 sm:bottom-6 sm:right-6 z-40 w-14 h-14 sm:w-16 sm:h-16 rounded-full shadow-2xl flex items-center justify-center transition-all duration-300 transform hover:scale-110 ${
        isOpen
          ? 'bg-red-600 hover:bg-red-700'
          : 'bg-gradient-to-r from-designColor to-[#c026d3] hover:shadow-designColor/50'
      }`}
      title={isOpen ? "Close chatbot" : "Ask Nagarjuna"}
    >
      {isOpen ? (
        <FaTimes className="text-white text-2xl" />
      ) : (
        <FaRobot className="text-white text-2xl animate-pulse" />
      )}
      {!isOpen && (
        <span className="absolute -top-1 -right-1 flex h-3 w-3">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#c026d3] opacity-75"></span>
          <span className="relative inline-flex rounded-full h-3 w-3 bg-purple-500"></span>
        </span>
      )}
    </button>
  );
};

export default ChatbotButton;
