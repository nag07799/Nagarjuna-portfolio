import React from 'react'
import { motion } from 'framer-motion';
import ResumeCard from './ResumeCard';

const Education = () => {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1, transition: { duration: 0.5 } }}
      className="w-full flex flex-col lgl:flex-row gap-10 lgl:gap-20"
    >
      {/* part one */}
      <div>
        <div className="py-6 lgl:py-12 font-titleFont flex flex-col gap-4">
          <p className="text-sm text-designColor tracking-[4px]">2014 - 2021</p>
          <h2 className="text-3xl md:text-4xl font-bold">Education Quality</h2>
        </div>
        <div className="mt-6 lgl:mt-14 w-full h-[1000px] border-l-[6px] border-l-black border-opacity-30 flex flex-col gap-10">
          <ResumeCard
            title="B.Tech in Electronics and Comminication Engg"
            subTitle="G. Pullareddy Engg college, Kurnool (2018-2021)"
            result="7.12/10"
            des="The training provided by universities in order to prepare people to work in various sectors of the economy or areas of culture."
          />
          <ResumeCard
            title="Diploma in Electronics and Comminication Engg"
            subTitle="GMR college, Srisailam (2015-2018)"
            result="84%"
            des="Higher education is tertiary education leading to award of an academic degree. Higher education, also called post-secondary education."
          />
          <ResumeCard
            title="Secondary School Education"
            subTitle="ZPHS, Kurnool (2014-2015) "
            result="7.7/10"
            des="Secondary education or post-primary education covers two phases on the International Standard Classification of Education scale."
          />
        </div>
      </div>
      {/* part Two */}

      <div>
        <div className="py-6 lgl:py-12 font-titleFont flex flex-col gap-4">
          <p className="text-sm text-designColor tracking-[4px]">. </p>
          <h2 className="text-3xl md:text-4xl font-bold">Certifications</h2>
        </div>
        <div className="mt-6 lgl:mt-14 w-full h-[1000px] border-l-[6px] border-l-black border-opacity-30 flex flex-col gap-10">
          <ResumeCard
            title="React JS(Redux & Router): The Complete guide"
            subTitle="Maximilian Schwarzmüller"
            result="Udemy"
            des="Hands-on experience in React, Redux, and React Router, mastering hooks, component 
            lifecycle, and building scalable, optimized, and modern web applications."
          />
          <ResumeCard
            title="JavaScript: The Complete Guide"
            subTitle="Maximilian Schwarzmüller"
            result="Udemy"
            des="Ddeep understanding of JavaScript fundamentals and advanced concepts, building dynamic, efficient, and interactive web applications with practical, hands-on experience."
          />
          <ResumeCard
            title="Git: The Practical Guide"
            subTitle="Maximilian Schwarzmüller"
            result="Udemy"
            des="Practical knowledge of Git, mastering version control, branching, merging, and repository management for effective collaboration and efficient project development."
          />
        </div>
      </div>
    </motion.div>
  );
}

export default Education