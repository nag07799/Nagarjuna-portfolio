import React, { useState } from "react";
import Banner from "./components/banner/Banner";
import Contact from "./components/contact/Contact";
import Features from "./components/features/Features";
import Footer from "./components/footer/Footer";
import FooterBottom from "./components/footer/FooterBottom";
import Navbar from "./components/navbar/Navbar";
import Projects from "./components/projects/Projects";
import Resume from "./components/resume/Resume";
import Chatbot from "./components/chatbot/Chatbot";
import ChatbotButton from "./components/chatbot/ChatbotButton";
// import Testimonial from "./components/tesimonial/Testimonial";

function App() {
  const [isChatbotOpen, setIsChatbotOpen] = useState(false);

  const toggleChatbot = () => {
    setIsChatbotOpen(!isChatbotOpen);
  };

  return (
    <div className="w-full h-auto bg-bodyColor text-lightText px-4">
        <Navbar onChatbotToggle={toggleChatbot} />
      <div className="max-w-screen-xl mx-auto">
        <Banner />
        <Features />
        <Projects />
        <Resume />
        {/* <Testimonial /> */}
        <Contact />
        <Footer />
        <FooterBottom />
      </div>
      <ChatbotButton isOpen={isChatbotOpen} onClick={toggleChatbot} />
      <Chatbot isOpen={isChatbotOpen} onClose={() => setIsChatbotOpen(false)} />
    </div>
  );
}

export default App;
