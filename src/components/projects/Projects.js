import React from 'react'
import Title from '../layouts/Title'
import { projectOne, projectTwo, projectThree } from "../../assets/index";
import ProjectsCard from './ProjectsCard';
// import { BsGithub } from 'react-icons/bs';
import { FaGithub } from 'react-icons/fa';

const Projects = () => {
  return (
    <section
      id="projects"
      className="w-full py-20 border-b-[1px] border-b-black"
    >
      <div className="flex justify-center items-center text-center">
        <Title
          title="AI/ML PRODUCTION SYSTEMS"
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
          title="Real-Time Fraud Detection System"
          des="Built production-ready fraud detection models processing 30M+ daily transactions with 98% precision. Implemented ensemble models combining XGBoost and deep neural networks with PySpark feature engineering, deployed on AWS SageMaker with autoscaling and MLflow tracking."
          src={projectOne}
          github={
            <a
              href='https://github.com/nagu0799'
              target='_blank'
              rel='noopener noreferrer'
              aria-label="GitHub repository"
            >
              {<FaGithub />}
            </a>
          }

        />
        <ProjectsCard
          title="Multi-Agent AI Banking Platform"
          des="Designed and implemented multi-agent AI platform using LangChain and LangGraph to automate trading, risk management, compliance, and customer analytics. Integrated Model Context Protocol (MCP) for seamless agent access to financial APIs and regulatory databases."
          src={projectTwo}

          github={
            <a
              href='https://github.com/nagu0799'
              target='_blank'
              rel='noopener noreferrer'
              aria-label="GitHub repository"
            >
              {<FaGithub />}
            </a>
          }
        />
        <ProjectsCard
          title="Agentic RAG Chatbot System"
          des="Built scalable Agentic RAG chatbot enabling 24/7 support for university students. Engineered agentic workflows with specialized sub-agents using Llama 3, Mistral, and vector databases (FAISS, Chroma) for semantic search and real-time student engagement."
          src={projectThree}
          github={
            <a
              href='https://github.com/nagu0799'
              target='_blank'
              rel='noopener noreferrer'
              aria-label="GitHub repository"
            >
              {<FaGithub />}
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