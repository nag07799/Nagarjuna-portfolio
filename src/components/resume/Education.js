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
          <p className="text-sm text-designColor tracking-[4px]">2023 - 2025</p>
          <h2 className="text-3xl md:text-4xl font-bold">Education Quality</h2>
        </div>
        <div className="mt-6 lgl:mt-14 w-full h-[1000px] border-l-[6px] border-l-black border-opacity-30 flex flex-col gap-10">
          <ResumeCard
            title="M.S. Data Science"
            subTitle="University of North Texas (Aug 2023 - May 2025)"
            result="4.0/4.0"
            des="Advanced graduate program specializing in machine learning, deep learning, statistical modeling, big data analytics, and AI systems. Focused on practical applications in real-world scenarios with hands-on projects in computer vision, NLP, and distributed computing."
          />
        </div>
      </div>
      {/* part Two */}

      <div>
        <div className="py-6 lgl:py-12 font-titleFont flex flex-col gap-4">
          <p className="text-sm text-designColor tracking-[4px]">Key Courses</p>
          <h2 className="text-3xl md:text-4xl font-bold">Specializations</h2>
        </div>
        <div className="mt-6 lgl:mt-14 w-full h-[1000px] border-l-[6px] border-l-black border-opacity-30 flex flex-col gap-10">
          <ResumeCard
            title="Machine Learning & Deep Learning"
            subTitle="Neural Networks, CNNs, RNNs, Transformers"
            result="Advanced"
            des="Comprehensive study of machine learning algorithms, deep neural networks, computer vision, and natural language processing. Hands-on experience with PyTorch, TensorFlow, and state-of-the-art architectures."
          />
          <ResumeCard
            title="MLOps & Production AI Systems"
            subTitle="AWS, Docker, Kubernetes, CI/CD"
            result="Advanced"
            des="End-to-end machine learning operations including model deployment, monitoring, versioning, and scaling. Expertise in building production-ready AI systems with cloud infrastructure and DevOps practices."
          />
          <ResumeCard
            title="Big Data & Distributed Computing"
            subTitle="Spark, PySpark, Distributed ML"
            result="Advanced"
            des="Large-scale data processing and distributed machine learning. Experience with Apache Spark, PySpark, and building scalable data pipelines for real-time and batch processing."
          />
        </div>
      </div>
    </motion.div>
  );
}

export default Education