import React from "react";
import {
  FaLinkedinIn,
  FaInstagram,
  FaDownload,
  FaGithub,
} from "react-icons/fa";
import { contactImg } from "../../assets/index";
import {Chinna} from '../../assets'
const ContactLeft = () => {
  // const chinna = "/Chinna.pdf";
  return (
    <div className="w-full lgl:w-[35%] h-full bg-gradient-to-r from-[#1e2024] to-[#23272b] p-4 lgl:p-8 rounded-lg shadow-shadowOne flex flex-col gap-8 justify-center">
      <img
        className="w-full h-64 object-cover rounded-lg mb-2"
        src={contactImg}
        alt="contactImg"
      />
      <div className="flex flex-col gap-4">
        <h3 className="text-3xl font-bold text-white">Chinna Kadinti</h3>
        <p className="text-lg font-normal text-gray-400">
          MERN Stack Developer
        </p>
        <p className="text-base text-gray-400 tracking-wide">
          Skilled in building responsive, user-friendly applications with
          MongoDB, Express.js, React, and Node.js. Passionate about performance,
          design, and creating seamless full-stack web experiences.{" "}
        </p>
        <p className="text-base text-gray-400 flex items-center gap-2">
          Phone: <span className="text-lightText">+91 9381287877</span>
        </p>
        <p className="text-base text-gray-400 flex items-center gap-2">
          Email:{" "}
          <span className="text-lightText">chinna.kadinti1@gmail.com</span>
        </p>
      </div>
      <div className="flex flex-col gap-4">
        <h2 className="text-base uppercase font-titleFont mb-4">Find me in</h2>
        <div className="flex gap-4">
          <span className="bannerIcon">
            <a
              href="https://www.linkedin.com/in/kadinti/"
              target="_blank"
              rel="noopener noreferrer"
            >
              <FaLinkedinIn />
            </a>
          </span>
          <span className="bannerIcon">
            <a
              href="https://github.com/Chinna-Kadinti/"
              target="_blank"
              rel="noopener noreferrer"
              aria-label="GitHub repository"
            >
              <FaGithub />
            </a>
          </span>

          <span className="bannerIcon">
            <a
              href="https://www.instagram.com/c_h_i_n_n_a.5/"
              target="_blank"
              rel="noopener noreferrer"
              aria-label="GitHub repository"
            >
              <FaInstagram />
            </a>
          </span>
        </div>
        <div>
          <h1 className="font-bold text-center text-red-500 text-base uppercase font-titleFont mb-4">
            RESUME
          </h1>
          <span className="bannerIcon w-full ">
            <a href={Chinna} rel="noopener noreferrer" target="_blank" download ={Chinna}>
              
                <FaDownload />
              
            </a>
          </span>
        </div>
      </div>
    </div>
  );
};

export default ContactLeft;
