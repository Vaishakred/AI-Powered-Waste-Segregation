import React, { useState } from "react";
import { HeroSection } from "@/components/HeroSection";
import { WasteClassifier } from "@/components/WasteClassifier";
import { ProjectOverview } from "@/components/ProjectOverview";
import { Navigation } from "@/components/Navigation";
import { ScrollToTop } from "@/components/ScrollToTop";

const Index = () => {
  const [currentSection, setCurrentSection] = useState<'hero' | 'classifier' | 'overview'>('hero');

  const handleSectionChange = (section: 'hero' | 'classifier' | 'overview') => {
    setCurrentSection(section);
  };

  const scrollToTop = () => {
    setCurrentSection('hero');
  };

  return (
    <div className="min-h-screen bg-background">
      <Navigation currentSection={currentSection} onSectionChange={handleSectionChange} />
      
      <main className="pt-20">
        {currentSection === 'hero' && (
          <HeroSection onGetStarted={() => handleSectionChange('classifier')} />
        )}
        {currentSection === 'classifier' && <WasteClassifier />}
        {currentSection === 'overview' && <ProjectOverview />}
      </main>

      <ScrollToTop onClick={scrollToTop} show={currentSection !== 'hero'} />

      {/* Footer */}
      <footer className="border-t border-primary/20 bg-background/80 backdrop-blur-sm">
        <div className="container mx-auto px-6 py-8">
          <div className="text-center text-muted-foreground">
            <p className="mb-2">
              © 2024 EcoClassify AI - 1M1B Internship Project
            </p>
            <p className="text-sm">
              Built for sustainable waste management with ❤️ and AI
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Index;
