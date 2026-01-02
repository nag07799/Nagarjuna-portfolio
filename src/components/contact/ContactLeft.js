import React from "react";
import {
  FaLinkedinIn,
  FaDownload,
  FaGithub,
} from "react-icons/fa";
import { contactImg } from "../../assets/index";
import { nagResume } from "../../assets";
const ContactLeft = () => {
  // const chinna = "/Chinna.pdf";
  return (
    <div className="w-full lgl:w-[35%] h-full bg-gradient-to-r from-[var(--color-panel-start)] to-[var(--color-panel-end)] p-4 lgl:p-8 rounded-lg shadow-shadowOne flex flex-col gap-8 justify-center">
      <img
        className="w-full h-64 object-cover rounded-lg mb-2"
        src={contactImg}
        alt="contactImg"
      />
      <div className="flex flex-col gap-4">
        <h3 className="text-3xl font-bold text-white">Nagarjuna Pendekanti</h3>
        <p className="text-lg font-normal text-gray-400">
          AI Engineer
        </p>
        <p className="text-base text-gray-400 tracking-wide">
          AI Engineer with 4+ years of experience building real-time financial AI, multi-agent systems, and production-ready ML solutions. Specialized in PyTorch, TensorFlow, LangChain, LangGraph, and end-to-end MLOps on AWS.{" "}
        </p>
        <p className="text-base text-gray-400 flex items-center gap-2">
          Phone: <span className="text-lightText">+1 (940) 843-8699</span>
        </p>
        <p className="text-base text-gray-400 flex items-center gap-2">
          Email:{" "}
          <span className="text-lightText">nagu07799@gmail.com</span>
        </p>
      </div>
      <div className="flex flex-col gap-4">
        <h2 className="text-base uppercase font-titleFont mb-4">Find me in</h2>
        <div className="flex gap-4">
          <span className="bannerIcon">
            <a
              href="https://www.linkedin.com/in/pendekanti/"
              target="_blank"
              rel="noopener noreferrer"
            >
              <FaLinkedinIn />
            </a>
          </span>
          <span className="bannerIcon">
            <a
              href="https://github.com/nag07799"
              target="_blank"
              rel="noopener noreferrer"
              aria-label="GitHub repository"
            >
              <FaGithub />
            </a>
          </span>
        </div>
        <div>
          <h1 className="font-bold text-center text-red-500 text-base uppercase font-titleFont mb-4">
            RESUME
          </h1>
          <span className="bannerIcon w-full ">
            <a
              href={nagResume}
              rel="noopener noreferrer"
              target="_blank"
              download={nagResume}
            >
              
                <FaDownload />
              
            </a>
          </span>
        </div>
      </div>
    </div>
  );
};

export default ContactLeft;
