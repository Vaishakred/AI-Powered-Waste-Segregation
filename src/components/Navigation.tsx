import React from "react";
import { Button } from "@/components/ui/button";
import { Recycle } from "lucide-react";

interface NavigationProps {
  currentSection: 'hero' | 'classifier' | 'overview';
  onSectionChange: (section: 'hero' | 'classifier' | 'overview') => void;
}

export function Navigation({ currentSection, onSectionChange }: NavigationProps) {
  const navItems = [
    { id: 'hero' as const, label: 'Home' },
    { id: 'classifier' as const, label: 'Classifier' },
    { id: 'overview' as const, label: 'Overview' },
  ];

  return (
    <nav className="fixed top-0 w-full bg-background/80 backdrop-blur-xl z-50 border-b border-primary/20">
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Logo */}
          <button
            onClick={() => onSectionChange('hero')}
            className="flex items-center gap-3 group"
          >
            <div className="w-10 h-10 bg-gradient-primary rounded-xl flex items-center justify-center group-hover:shadow-glow transition-all duration-300">
              <Recycle className="w-5 h-5 text-primary-foreground" />
            </div>
            <span className="text-xl font-bold bg-gradient-to-r from-primary-glow to-primary bg-clip-text text-transparent">
              EcoClassify AI
            </span>
          </button>

          {/* Navigation Items */}
          <div className="hidden md:flex items-center gap-8">
            {navItems.map((item) => (
              <Button
                key={item.id}
                variant="ghost"
                onClick={() => onSectionChange(item.id)}
                className={`
                  relative transition-all duration-300
                  ${currentSection === item.id 
                    ? 'text-primary-glow' 
                    : 'text-muted-foreground hover:text-foreground'
                  }
                `}
              >
                {item.label}
                {currentSection === item.id && (
                  <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-primary rounded-full" />
                )}
              </Button>
            ))}
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden">
            <Button variant="ghost" size="sm">
              Menu
            </Button>
          </div>
        </div>
      </div>
    </nav>
  );
}