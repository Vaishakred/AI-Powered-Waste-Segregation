import React from "react";
import { ArrowUp } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ScrollToTopProps {
  onClick: () => void;
  show: boolean;
}

export function ScrollToTop({ onClick, show }: ScrollToTopProps) {
  if (!show) return null;

  return (
    <Button
      onClick={onClick}
      size="icon"
      className="
        fixed bottom-8 right-8 w-12 h-12 
        bg-gradient-primary hover:shadow-glow 
        shadow-elegant transition-all duration-300
        hover:scale-110 pulse-glow
      "
    >
      <ArrowUp className="w-5 h-5" />
    </Button>
  );
}