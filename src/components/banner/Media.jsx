import React from 'react'
import { FaPython, FaAws, FaDocker } from "react-icons/fa";
import { SiPytorch, SiTensorflow, SiKubernetes } from "react-icons/si";
const Media = () => {
  return (
    <div className="flex flex-col xl:flex-row gap-6 lgl:gap-0 justify-between">
        {/* <div>
          <h2 className="text-base uppercase font-titleFont mb-4">
            Find me in
          </h2>
          <div className="flex gap-4">
            <span className="bannerIcon">
              <FaFacebookF />
            </span>
            <span className="bannerIcon">
              <FaTwitter />
            </span>
            <span className="bannerIcon">
              <FaLinkedinIn />
            </span>
          </div>
        </div> */}
        <div>
          <h2 className="text-base uppercase font-titleFont mb-4">
            BEST SKILL ON
          </h2>
          <div className="flex gap-4">
          <span className="bannerIcon">
              <FaPython />
            </span>
            <span className="bannerIcon">
              <SiPytorch />
            </span>
            <span className="bannerIcon">
              <SiTensorflow />
            </span>
            <span className="bannerIcon">
              <FaAws />
            </span>
            <span className="bannerIcon">
              <FaDocker />
            </span>
            <span className="bannerIcon">
              <SiKubernetes />
            </span>

          </div>
        </div>
      </div>
  )
}

export default Media