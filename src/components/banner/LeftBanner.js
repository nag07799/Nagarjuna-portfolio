import React from 'react'
import { useTypewriter, Cursor } from "react-simple-typewriter";
import Media from './Media';

const LeftBanner = () => {
  const [text] = useTypewriter({
    words: ["AI Engineer.", "ML Engineer.", "Multi-Agent Systems Developer.", "MLOps Specialist."],
    loop: true,
    typeSpeed: 50,
    deleteSpeed: 20,
    delaySpeed: 2000,
  });
  return (
    <div className="w-full lgl:w-1/2 flex flex-col gap-20">
      <div className="flex flex-col gap-5">
        <h4 className="text-sm sm:text-base md:text-lg font-normal">WELCOME TO MY PORTFOLIO</h4>
        <h1 className="text-2xl sm:text-3xl md:text-4xl lg:text-5xl font-bold text-white">
          Hi, I'm <span className="text-designColor capitalize">Nagarjuna Pendekanti</span>
        </h1>
        <h2 className="text-xl sm:text-2xl md:text-3xl font-bold text-white">
          <span>{text}</span>
          <Cursor
            cursorBlinking="false"
            cursorStyle="|"
            cursorColor="#ff014f"
          />
        </h2>
        <p className="text-sm sm:text-base font-bodyFont leading-6 tracking-wide">
        AI Engineer with 4+ years building real-time financial AI and multi-agent systems. I specialize in developing production-ready AI solutions using PyTorch, TensorFlow, LangChain, and LangGraph. My expertise spans fraud detection at scale, agentic platforms with Model Context Protocol, and end-to-end MLOps with Docker, Kubernetes, and AWS. I deliver intelligent systems that combine cutting-edge AI with robust engineering practices to solve complex real-world problems.        </p>
      </div>
      {/* Media */}
      <Media />
    </div>
  );
}

export default LeftBanner