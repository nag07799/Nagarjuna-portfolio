import React from 'react'
import Title from '../layouts/Title'
import { projectThree } from "../../assets/index";
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
      <div className="flex justify-center items-center px-4">
        <div className="w-full max-w-sm sm:max-w-md md:max-w-lg lg:max-w-xl">
          <ProjectsCard
            title="Agentic RAG Chatbot System"
            des="Production-ready Retrieval-Augmented Generation system with agentic reasoning powered by Ollama and Mistral. Features local-first architecture, custom NumPy vector database, hybrid search (semantic + BM25), ReAct reasoning pattern, and hallucination prevention with zero API costs."
            src={projectThree}
            github={
              <a
                href='https://github.com/nag07799/agentic-rag'
                target='_blank'
                rel='noopener noreferrer'
                aria-label="GitHub repository"
              >
                <FaGithub />
              </a>
            }
          />
        </div>
      </div>
    </section>
  );
}

export default Projects