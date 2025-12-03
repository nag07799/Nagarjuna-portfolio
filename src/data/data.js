import { FaBrain, FaRobot, FaServer, FaChartLine, FaShieldAlt } from 'react-icons/fa';
import { SiKubernetes } from 'react-icons/si';

// Features Data
export const featuresData = [
  {
    id: 1,
    icon: <FaBrain/>,
    title: "Deep Learning & Neural Networks",
    des: "Building state-of-the-art deep learning models with PyTorch and TensorFlow, specializing in CNNs, RNNs, and Transformers for computer vision and NLP applications.",
  },
  {
    id: 2,
    icon: <FaRobot />,
    title: "Multi-Agent AI Systems",
    des: "Architecting intelligent agent systems using LangChain, LangGraph, and Model Context Protocol (MCP) for collaborative decision-making and autonomous workflows.",
  },
  {
    id: 3,
    icon: <FaServer />,
    title: "Production ML Infrastructure",
    des: "Deploying scalable ML systems on AWS (SageMaker, Bedrock, Lambda) with Docker, Kubernetes, and comprehensive MLOps pipelines for enterprise-grade reliability.",
  },
  {
    id: 4,
    icon: <FaChartLine />,
    title: "Real-Time ML Systems",
    des: "Engineering high-throughput ML pipelines processing millions of events daily with PySpark, Kafka, and streaming architectures for instant predictions.",
  },
  {
    id: 5,
    icon: <SiKubernetes />,
    title: "MLOps & CI/CD Automation",
    des: "Implementing end-to-end MLOps with MLflow, GitHub Actions, Terraform, and automated testing to accelerate model deployment from hours to minutes.",
  },
  {
    id: 6,
    icon: <FaShieldAlt />,
    title: "AI Governance & Explainability",
    des: "Delivering interpretable AI with SHAP, model drift monitoring, and compliance-ready documentation for regulatory audits and stakeholder trust.",
  },
];
