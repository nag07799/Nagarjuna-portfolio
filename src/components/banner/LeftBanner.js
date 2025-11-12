import React from 'react'
import { useTypewriter, Cursor } from "react-simple-typewriter";
import Media from './Media';

const LeftBanner = () => {
  const [text] = useTypewriter({
    words: ["Professional Coder.", "Fullstack Developer.", "Frontend Expert.", "Backend Developer."],
    loop: true,
    typeSpeed: 50,
    deleteSpeed: 20,
    delaySpeed: 2000,
  });
  return (
    <div className="w-full lgl:w-1/2 flex flex-col gap-20">
      <div className="flex flex-col gap-5">
        <h4 className=" text-lg font-normal">WELCOME TO MY PORTFOLIO</h4>
        <h1 className="text-5xl font-bold text-white">
          Hi, I'm <span className="text-designColor capitalize">Chinna Kadinti</span>
        </h1>
        <h2 className="text-3xl font-bold text-white">
          <span>{text}</span>
          <Cursor
            cursorBlinking="false"
            cursorStyle="|"
            cursorColor="#ff014f"
          />
        </h2>
        <p className="text-base font-bodyFont leading-6 tracking-wide">
        I'm a Fullstack developer passionate about turning ideas into engaging digital experiences. I specialize in building intuitive, responsive user interfaces that blend functionality with aesthetic appeal. My approach centers on understanding user needs and translating them into seamless, accessible designs that feel natural and delightful. Whether it's crafting pixel-perfect layouts or optimizing performance, I’m committed to delivering code that brings designs to life and provides users with an exceptional journey.        </p>
      </div>
      {/* Media */}
      <Media />
    </div>
  );
}

export default LeftBanner