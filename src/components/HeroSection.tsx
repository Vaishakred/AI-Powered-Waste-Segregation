import React from "react";
import { ArrowRight, Target, Recycle, Globe, Zap, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import heroImage from "@/assets/hero-image.jpg";

const heroStats = [
  { label: "Accuracy Rate", value: "94.2%", icon: Target },
  { label: "Items Classified", value: "50K+", icon: Recycle },
  { label: "CO₂ Reduced", value: "12.5T", icon: Globe },
  { label: "Processing Speed", value: "< 2s", icon: Zap },
];

interface HeroSectionProps {
  onGetStarted: () => void;
}

export function HeroSection({ onGetStarted }: HeroSectionProps) {
  return (
    <div className="min-h-screen bg-animated hero-pattern relative overflow-hidden">
      {/* Floating particles effect */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {[...Array(20)].map((_, i) => (
          <div
            key={i}
            className="absolute w-2 h-2 bg-primary-glow/20 rounded-full float"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 6}s`,
              animationDuration: `${6 + Math.random() * 3}s`,
            }}
          />
        ))}
      </div>

      <div className="relative container mx-auto px-6 py-20">
        <div className="grid lg:grid-cols-2 gap-12 items-center min-h-[80vh]">
          {/* Left Content */}
          <div className="space-y-8">
            <div className="space-y-6">
              <div className="inline-flex items-center gap-2 px-4 py-2 bg-primary/10 border border-primary/20 rounded-full text-sm text-primary-glow">
                <Sparkles className="w-4 h-4" />
                AI-Powered Classification
              </div>
              
              <h1 className="text-5xl lg:text-7xl font-bold leading-tight">
                Smart Waste
                <span className="block text-primary-glow glow-text">
                  Segregation
                </span>
              </h1>
              
              <p className="text-xl text-muted-foreground max-w-lg leading-relaxed">
                Automatically classify cardboard, glass, metal, paper, plastic, and trash with 94%+ accuracy using advanced computer vision.
              </p>
            </div>

            <div className="flex flex-wrap gap-4">
              <Button 
                onClick={onGetStarted}
                size="lg"
                className="bg-gradient-primary hover:shadow-glow transition-all duration-300 group"
              >
                Get Started
                <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </Button>
              
              <Button 
                variant="outline" 
                size="lg"
                className="border-primary/30 text-primary-glow hover:bg-primary/10"
              >
                View Demo
              </Button>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-8">
              {heroStats.map((stat, index) => (
                <Card 
                  key={stat.label} 
                  className="p-4 bg-gradient-card border-primary/20 glow-card group hover:border-primary/40 transition-all duration-300"
                  style={{ animationDelay: `${index * 100}ms` }}
                >
                  <div className="text-center space-y-2">
                    <div className="mx-auto w-10 h-10 bg-primary/20 rounded-lg flex items-center justify-center group-hover:bg-primary/30 transition-colors">
                      <stat.icon className="w-5 h-5 text-primary-glow" />
                    </div>
                    <div className="text-2xl font-bold text-foreground">{stat.value}</div>
                    <div className="text-sm text-muted-foreground">{stat.label}</div>
                  </div>
                </Card>
              ))}
            </div>
          </div>

          {/* Right Content - Hero Image */}
          <div className="relative">
            <div className="relative group">
              <div className="absolute -inset-4 bg-gradient-primary opacity-20 blur-xl group-hover:opacity-30 transition-opacity duration-300"></div>
              <img 
                src={heroImage} 
                alt="AI-powered waste sorting facility with robotic arms and neural networks"
                className="relative rounded-2xl shadow-elegant w-full h-auto float"
              />
              
              {/* Floating badge */}
              <div className="absolute -bottom-4 -left-4 bg-gradient-card border border-primary/30 rounded-xl p-4 shadow-elegant">
                <div className="flex items-center gap-3">
                  <div className="w-3 h-3 bg-success rounded-full pulse-glow"></div>
                  <div>
                    <div className="text-sm font-semibold">Live Classification</div>
                    <div className="text-xs text-muted-foreground">99.2% Uptime</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}