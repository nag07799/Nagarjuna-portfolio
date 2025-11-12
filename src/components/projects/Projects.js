import React from 'react'
import Title from '../layouts/Title'
import { projectOne, projectTwo, projectThree } from "../../assets/index";
import ProjectsCard from './ProjectsCard';
// import { BsGithub } from 'react-icons/bs';
import { FaGithub, FaGlobe } from 'react-icons/fa';

const Projects = () => {
  return (
    <section
      id="projects"
      className="w-full py-20 border-b-[1px] border-b-black"
    >
      <div className="flex justify-center items-center text-center">
        <Title
          title="VISIT MY PORTFOLIO AND KEEP YOUR FEEDBACK"
          des="My Projects"
        />
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6 xl:gap-14">
        {/* <ProjectsCard
          title="SOCIAL MEDIA CLONE"
          des=" Lorem, ipsum dolor sit amet consectetur adipisicing elit.
              Explicabo quibusdam voluptate sapiente voluptatibus harum quidem!"
          src={projectOne}
          github={<a href='https://ytapi-clone-chinna.web.app' target='_blank'><BsGithub /></a>}
        /> */}
        <ProjectsCard
          title="YOUTUBE CLONE"
          des="A ReactJS-based YouTube clone that integrates with the YouTube API. Users 
            can search for and watch videos, access channel information, and enjoy a 
            responsive design for desktop and mobile."
          src={projectOne}
          github={
            <a 
              href='https://github.com/Chinna-Kadinti/youtube-clone.git' 
              target='_blank' 
              rel='noopener noreferrer'
              aria-label="GitHub repository"
            >
              {<FaGithub />}
            </a>
          }
          globe={
            <a 
              href='https://ytapi-clone-chinna.web.app' 
              target='_blank' 
              rel='noopener noreferrer'
              aria-label="Globe"
            >
              {<FaGlobe />}
            </a>
          }
          
        />
        <ProjectsCard
          title="NETFLIX CLONE"
          des=" A React-based Netflix clone that uses CSS for styling and integrates with the TMDB API to display live
           movie listings. Users can browse and view details of popular films in a responsive design."
          src={projectTwo}
          
          github={
            <a 
              href='https://github.com/Chinna-Kadinti/netflix-clone-chinna.git' 
              target='_blank' 
              rel='noopener noreferrer'
              aria-label="GitHub repository"
            >
              {<FaGithub />}
            </a>
          }
          globe={
            <a 
              href='https://netflix-clone-chinna.web.app/' 
              target='_blank' 
              rel='noopener noreferrer'
              aria-label="Globe"
            >
              {<FaGlobe />}
            </a>
          }
        />
        <ProjectsCard
          title="Personal porfolio"
          des=" A personal portfolio showcasing my skills, projects, and experiences. It features information about me, my work, a
          nd various functionalities to highlight my capabilities and accomplishments effectively."
          src={projectThree}
          github={
            <a 
              href='https://github.com/Chinna-Kadinti/chinna-kadinti.git' 
              target='_blank' 
              rel='noopener noreferrer'
              aria-label="GitHub repository"
            >
              {<FaGithub />}
            </a>
          }
          globe={
            <a 
              href='https://chinna-kadinti.netlify.app/' 
              target='_blank' 
              rel='noopener noreferrer'
              aria-label="Globe"
            >
              {<FaGlobe />}
            </a>
          }
        />
        {/* <ProjectsCard
          title="SOCIAL MEDIA CLONE"
          des=" Lorem, ipsum dolor sit amet consectetur adipisicing elit.
              Explicabo quibusdam voluptate sapiente voluptatibus harum quidem!"
          src={projectThree}
        /> */}
        {/* <ProjectsCard
          title="E-commerce Website"
          des=" Lorem, ipsum dolor sit amet consectetur adipisicing elit.
              Explicabo quibusdam voluptate sapiente voluptatibus harum quidem!"
          src={projectOne}
        />
        <ProjectsCard
          title="Chatting App"
          des=" Lorem, ipsum dolor sit amet consectetur adipisicing elit.
              Explicabo quibusdam voluptate sapiente voluptatibus harum quidem!"
          src={projectTwo}
        /> */}
      </div>
    </section>
  );
}

export default Projects