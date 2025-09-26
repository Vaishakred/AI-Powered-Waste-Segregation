import React from "react";
import { Database, Brain, Globe, Code, CheckCircle, Target, Users, Zap } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const features = [
  { 
    icon: Database, 
    title: "Dataset Acquisition", 
    description: "Utilized a comprehensive Kaggle dataset with thousands of high-quality images across 6 waste categories for robust training.", 
    status: "completed" 
  },
  { 
    icon: Brain, 
    title: "Model Training", 
    description: "TensorFlow/Keras transfer learning model based on MobileNetV2 architecture achieving 94%+ accuracy through advanced data augmentation.", 
    status: "completed" 
  },
  { 
    icon: Globe, 
    title: "Web Deployment", 
    description: "Modern React/TypeScript frontend with Python/Flask backend API for real-time image classification and seamless user experience.", 
    status: "completed" 
  }
];

const techStack = [
  { name: "TensorFlow", category: "ML" },
  { name: "Keras", category: "ML" },
  { name: "Python", category: "Backend" },
  { name: "Flask", category: "Backend" },
  { name: "React", category: "Frontend" },
  { name: "TypeScript", category: "Frontend" },
  { name: "Vite", category: "Frontend" },
  { name: "Tailwind CSS", category: "Frontend" }
];

const achievements = [
  { icon: Target, label: "Model Accuracy", value: "94.2%", color: "text-primary-glow" },
  { icon: Users, label: "Images Processed", value: "50K+", color: "text-blue-400" },
  { icon: Zap, label: "Avg Response Time", value: "< 2s", color: "text-yellow-400" },
  { icon: Globe, label: "CO₂ Impact", value: "12.5T", color: "text-green-400" },
];

export function ProjectOverview() {
  return (
    <div className="container mx-auto px-6 py-16">
      <div className="text-center mb-16">
        <div className="inline-flex items-center gap-2 px-4 py-2 bg-primary/10 border border-primary/20 rounded-full text-sm text-primary-glow mb-4">
          <Code className="w-4 h-4" />
          Project Deep Dive
        </div>
        <h2 className="text-4xl font-bold mb-6">Project Overview</h2>
        <p className="text-xl text-muted-foreground max-w-4xl mx-auto leading-relaxed">
          An advanced AI-powered waste segregation system that leverages computer vision and deep learning 
          to automatically classify materials into six categories: cardboard, glass, metal, paper, plastic, and trash.
        </p>
      </div>

      {/* Achievements Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-16">
        {achievements.map((achievement, index) => (
          <Card 
            key={achievement.label}
            className="p-6 text-center bg-gradient-card border-primary/20 glow-card hover:border-primary/40 transition-all duration-300"
          >
            <div className={`w-12 h-12 mx-auto mb-4 bg-black/20 rounded-xl flex items-center justify-center`}>
              <achievement.icon className={`w-6 h-6 ${achievement.color}`} />
            </div>
            <div className="text-2xl font-bold mb-1">{achievement.value}</div>
            <div className="text-sm text-muted-foreground">{achievement.label}</div>
          </Card>
        ))}
      </div>

      {/* Features Grid */}
      <div className="grid md:grid-cols-3 gap-8 mb-16">
        {features.map((feature, index) => (
          <Card 
            key={feature.title}
            className="bg-gradient-card border-primary/20 glow-card hover:border-primary/40 transition-all duration-300 overflow-hidden"
          >
            <div className="p-6 border-b border-primary/10">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-primary/20 rounded-xl flex items-center justify-center">
                    <feature.icon className="w-6 h-6 text-primary-glow" />
                  </div>
                  <h3 className="text-xl font-semibold">{feature.title}</h3>
                </div>
                <Badge variant="secondary" className="bg-success/20 text-success">
                  <CheckCircle className="w-3 h-3 mr-1" />
                  {feature.status}
                </Badge>
              </div>
            </div>
            
            <div className="p-6">
              <p className="text-muted-foreground leading-relaxed">{feature.description}</p>
            </div>
          </Card>
        ))}
      </div>

      {/* Technology Stack */}
      <Card className="p-8 bg-gradient-card border-primary/20 glow-card">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-12 h-12 bg-primary/20 rounded-xl flex items-center justify-center">
            <Code className="w-6 h-6 text-primary-glow" />
          </div>
          <h3 className="text-2xl font-semibold">Technology Stack</h3>
        </div>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {techStack.map((tech, index) => (
            <div key={tech.name} className="group">
              <div className="bg-muted/30 hover:bg-primary/10 border border-primary/20 rounded-lg p-4 text-center transition-all duration-300 hover:border-primary/40">
                <div className="font-medium text-foreground">{tech.name}</div>
                <div className="text-xs text-muted-foreground mt-1">{tech.category}</div>
              </div>
            </div>
          ))}
        </div>
        
        <div className="mt-8 p-6 bg-muted/20 rounded-xl border border-primary/10">
          <h4 className="font-semibold text-primary-glow mb-3">Key Features & Benefits:</h4>
          <div className="grid md:grid-cols-2 gap-4 text-sm text-muted-foreground">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <span className="text-primary-glow">•</span>
                Real-time image classification with sub-2 second response times
              </div>
              <div className="flex items-center gap-2">
                <span className="text-primary-glow">•</span>
                Mobile-optimized MobileNetV2 architecture for efficiency
              </div>
              <div className="flex items-center gap-2">
                <span className="text-primary-glow">•</span>
                Comprehensive recycling recommendations for each category
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <span className="text-primary-glow">•</span>
                Responsive design optimized for mobile and desktop
              </div>
              <div className="flex items-center gap-2">
                <span className="text-primary-glow">•</span>
                Scalable cloud deployment with RESTful API architecture
              </div>
              <div className="flex items-center gap-2">
                <span className="text-primary-glow">•</span>
                Environmental impact tracking and sustainability metrics
              </div>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}