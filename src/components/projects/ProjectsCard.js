import React from 'react'
// import { BsGithub } from "react-icons/bs";
// import { FaGlobe } from "react-icons/fa";

const ProjectsCard = ({ title, des, src, github, globe }) => {
  return (
    <div className="w-full p-4 sm:p-6 md:p-8 lg:px-10 lg:py-8 xl:px-12 xl:py-10 rounded-lg shadow-shadowOne flex flex-col bg-gradient-to-r from-bodyColor to-[var(--color-card-end)] group hover:bg-gradient-to-b hover:from-gray-900 hover:to-[var(--color-panel-start)] transition-colors duration-1000">
      <div className="w-full h-[80%] overflow-hidden rounded-lg">
        <img
          className="w-full h-48 sm:h-52 md:h-56 lg:h-60 object-cover group-hover:scale-110 duration-300 cursor-pointer"
          src={src}
          alt="src"
        />
      </div>
      <div className="w-full mt-5 flex flex-col  gap-6">
        <div>
          <div className="flex items-center justify-between">
            <h3 className="text-base uppercase text-designColor font-normal">
              {title}
            </h3>
            <div className="flex gap-2">
              <span className="text-lg w-10 h-10 rounded-full bg-black inline-flex justify-center items-center text-gray-400 hover:text-designColor duration-300 cursor-pointer">
                {github}
              </span>
              <span className="text-lg w-10 h-10 rounded-full bg-black inline-flex justify-center items-center text-gray-400 hover:text-designColor duration-300 cursor-pointer">
               {globe} 
              </span>
            </div>
          </div>
          <p className="text-sm tracking-wide mt-3 hover:text-gray-100 duration-300">
            {des}
          </p>
        </div>
      </div>
    </div>
  );
}

export default ProjectsCard
