import React from "react";
import { motion } from "framer-motion";
import ResumeCard from "./ResumeCard";

const Experience = () => {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1, transition: { duration: 0.5 } }}
      className="py-12 font-titleFont flex gap-20"
    >
      <div>
        <div className="flex flex-col gap-4">
          
          <h2 className="text-4xl font-bold">Job Experience</h2>
        </div>
        <div className="mt-14 w-full h-[px] border-l-[6px] border-l-black border-opacity-30 flex flex-col gap-10">
          <ResumeCard
            title="AI Engineer"
            subTitle="Procal Technologies (Client: Wells Fargo) - (June 2025 - Present)"
            result="Remote"
            des={<div><ul >
               <li>Designed and implemented multi-agent AI platform using LangChain and LangGraph to automate trading, risk management, compliance, and customer analytics for large-scale banking operations.</li>
           <li>Architected specialized agent workflows with Trading Agent, Risk Agent, Compliance Agent, and Customer Intelligence Agent modules for collaborative decision-making.</li>
           <li>Integrated Model Context Protocol (MCP) for seamless agent access to financial APIs, regulatory databases, live market feeds, and AML screening systems.</li>
           <li>Developed agent memory systems combining real-time context with persistent knowledge bases, ensuring continuity across 1,000+ user sessions.</li></ul></div>}
          />
          <ResumeCard
            title="ML Engineer"
            subTitle="Tata Consultancy Services (Client: Bank of America) - (Aug 2021 - Aug 2023)"
            result="Remote"
            des={<div><ul >
               <li>Built real-time fraud detection models for 30M+ daily transactions, improving precision to 98% and reducing false positives by 35%.</li>
           <li>Engineered streaming feature store with PySpark, delivering 200+ behavioral and merchant features with sub-second latency.</li>
           <li>Deployed PyTorch and TensorFlow models on AWS SageMaker with autoscaling, maintaining 99.9% uptime across peak loads.</li>
           <li>Implemented end-to-end ML CI/CD with GitHub Actions, Docker, Terraform, and Helm, reducing deployment cycles from days to under 3 hours.</li>
           <li>Delivered SHAP-based explainability and transaction-level reason codes, meeting governance requirements for regulator and internal audit reviews.</li></ul></div>}
          />
          <ResumeCard
            title="Research Assistant"
            subTitle="University of North Texas - (Jan 2024 - May 2025)"
            result="Denton, TX"
            des={<div><ul >
               <li>Designed and implemented Agentic RAG chatbot enabling 24/7 support for UNT students on library access, scholarships, campus services, and academic queries.</li>
           <li>Engineered agentic workflows integrating specialized sub-agents (Library Info Agent, Scholarship Advisor, Campus Services Agent) for domain-specific reasoning and multi-turn dialogues.</li>
           <li>Leveraged open-source LLMs (Llama 3, Mistral) and vector databases (FAISS, Chroma) for semantic search and rapid retrieval from diverse university data sources.</li>
           <li>Built scalable, user-friendly web interface with contextual chat history, document uploads, and interactive Q&A tailored for real-time student engagement.</li></ul></div>}
          />
          {/* <ResumeCard
            title="Web Developer & Trainer"
            subTitle="Apple Developer Team - (2012 - 2016)"
            result="MALAYSIA"
            des="A popular destination with a growing number of highly qualified homegrown graduates, it's true that securing a role in Malaysia isn't easy."
          />
          <ResumeCard
            title="Front-end Developer"
            subTitle="Nike - (2020 - 2011)"
            result="Oman"
            des="The Oman economy has grown strongly over recent years, having transformed itself from a producer and innovation-based economy."
          /> */}
        </div>
      </div>
      {/* <div>
        <div className="flex flex-col gap-4">
          <p className="text-sm text-designColor tracking-[4px]">2001 - 2020</p>
          <h2 className="text-4xl font-bold">Trainer Experience</h2>
        </div>
        <div className="mt-14 w-full h-[1000px] border-l-[6px] border-l-black border-opacity-30 flex flex-col gap-10">
          <ResumeCard
            title="Gym Instructor"
            subTitle="Rainbow Gym Center (2015 - 2020)"
            result="DHAKA"
            des="The training provided by universities in order to prepare people to work in various sectors of the economy or areas of culture."
          />
          <ResumeCard
            title="Web Developer and Instructor"
            subTitle="SuperKing College (2010 - 2014)"
            result="CANADA"
            des="Higher education is tertiary education leading to award of an academic degree. Higher education, also called post-secondary education."
          />
          <ResumeCard
            title="School Teacher"
            subTitle="Kingstar Secondary School (2001 - 2010)"
            result="NEVADA"
            des="Secondary education or post-primary education covers two phases on the International Standard Classification of Education scale."
          />
        </div> */}
      {/* </div> */}
    </motion.div>
  );
};

export default Experience;
