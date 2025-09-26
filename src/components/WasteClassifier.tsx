import React, { useState, useCallback } from "react";
import { Camera, Upload, AlertCircle, CheckCircle, Brain, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";

interface ClassificationResult {
  category: string;
  confidence: number;
  icon: React.ReactNode;
  recommendations: string[];
}

const wasteCategories: { [key: string]: { icon: string; name: string; color: string; recommendations: string[] } } = {
  cardboard: { 
    icon: "📦", 
    name: "Cardboard", 
    color: "bg-amber-500/20 text-amber-500",
    recommendations: ["Flatten boxes before recycling", "Remove any plastic tape or labels", "Keep it dry"] 
  },
  glass: { 
    icon: "🍾", 
    name: "Glass", 
    color: "bg-blue-500/20 text-blue-500",
    recommendations: ["Rinse bottles and jars", "Remove lids", "Do not include broken glass"] 
  },
  metal: { 
    icon: "⚙️", 
    name: "Metal", 
    color: "bg-slate-500/20 text-slate-300",
    recommendations: ["Rinse cans", "Aluminum and steel cans are highly recyclable", "Do not include electronics"] 
  },
  paper: { 
    icon: "📄", 
    name: "Paper", 
    color: "bg-green-500/20 text-green-500",
    recommendations: ["Remove bindings", "Keep paper clean and dry", "Shredded paper should be bagged"] 
  },
  plastic: { 
    icon: "💧", 
    name: "Plastic", 
    color: "bg-cyan-500/20 text-cyan-500",
    recommendations: ["Rinse containers", "Check the recycling number (1 and 2 are common)", "Remove caps"] 
  },
  trash: { 
    icon: "🗑️", 
    name: "Trash", 
    color: "bg-red-500/20 text-red-500",
    recommendations: ["This item is likely not recyclable", "Dispose of in a general waste bin", "Includes styrofoam, soiled food containers"] 
  },
};

export function WasteClassifier() {
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [isClassifying, setIsClassifying] = useState(false);
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const { toast } = useToast();

  const handleImageUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (file.size > 10 * 1024 * 1024) { // 10MB limit
        toast({
          title: "File too large",
          description: "Please select an image smaller than 10MB",
          variant: "destructive",
        });
        return;
      }
      
      setImageFile(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target?.result as string);
        setResult(null);
      };
      reader.readAsDataURL(file);
    }
  }, [toast]);

  // --- THIS IS THE ONLY PART THAT HAS CHANGED ---
  // It now makes a real API call to your Python backend.
  const classifyWaste = useCallback(async () => {
    if (!imageFile) return;
    
    setIsClassifying(true);
    setResult(null);

    const apiUrl = 'http://127.0.0.1:5000/predict';
    const formData = new FormData();
    formData.append('file', imageFile);

    try {
      // 1. Send the image to the live backend server
      const response = await fetch(apiUrl, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Server returned an error' }));
        throw new Error(`Server error: ${response.status}. ${errorData.error || ''}`);
      }

      // 2. Get the real prediction from your AI model
      const backendResult = await response.json();
      const categoryKey = backendResult.prediction?.toLowerCase() as keyof typeof wasteCategories;
      const categoryInfo = wasteCategories[categoryKey];

      if (categoryInfo) {
        // 3. Update the UI with the real result
        const finalResult: ClassificationResult = {
          category: categoryInfo.name,
          confidence: Math.round(backendResult.confidence * 100),
          icon: categoryInfo.icon,
          recommendations: categoryInfo.recommendations,
        };
        
        setResult(finalResult);
        
        toast({
          title: "Classification Complete!",
          description: `Identified as ${categoryInfo.name} with ${finalResult.confidence}% confidence`,
        });
      } else {
        throw new Error(`Unknown category received: ${backendResult.prediction}`);
      }
    } catch (error) {
      console.error("Classification failed:", error);
      toast({
        title: "Classification Failed",
        description: "Could not connect to the AI server. Please ensure it is running.",
        variant: "destructive",
      });
    } finally {
      setIsClassifying(false);
    }
  }, [imageFile, toast]);

  const clearImage = () => {
    setImagePreview(null);
    setImageFile(null);
    setResult(null);
  };

  // The rest of your beautiful UI code remains unchanged
  return (
    <div className="container mx-auto px-6 py-16">
      <div className="text-center mb-12">
        <div className="inline-flex items-center gap-2 px-4 py-2 bg-primary/10 border border-primary/20 rounded-full text-sm text-primary-glow mb-4">
          <Brain className="w-4 h-4" />
          AI Classification Engine
        </div>
        <h2 className="text-4xl font-bold mb-4">Waste Classifier</h2>
        <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
          Upload an image to get instant, accurate classification with recycling recommendations
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-8 max-w-6xl mx-auto">
        {/* Upload Section */}
        <Card className="p-8 bg-gradient-card border-primary/20 glow-card">
          <div className="space-y-6">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-primary/20 rounded-lg flex items-center justify-center">
                <Camera className="w-5 h-5 text-primary-glow" />
              </div>
              <h3 className="text-xl font-semibold">Upload Image</h3>
            </div>

            <div className="relative">
              {imagePreview ? (
                <div className="space-y-4">
                  <div className="relative group">
                    <img 
                      src={imagePreview} 
                      alt="Selected waste item" 
                      className="w-full h-64 object-cover rounded-xl shadow-card"
                    />
                    <div className="absolute inset-0 bg-black/20 opacity-0 group-hover:opacity-100 transition-opacity rounded-xl flex items-center justify-center">
                      <Button
                        onClick={clearImage}
                        variant="secondary"
                        size="sm"
                      >
                        Clear Image
                      </Button>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="border-2 border-dashed border-primary/30 rounded-xl p-12 text-center hover:border-primary/50 transition-colors">
                  <div className="space-y-4">
                    <div className="mx-auto w-16 h-16 bg-primary/10 rounded-2xl flex items-center justify-center">
                      <Upload className="w-8 h-8 text-primary-glow" />
                    </div>
                    <div>
                      <p className="text-lg font-medium">Drop your image here</p>
                      <p className="text-muted-foreground">or click to browse</p>
                    </div>
                  </div>
                </div>
              )}
              
              <input 
                type="file" 
                accept="image/*" 
                onChange={handleImageUpload}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
            </div>

            <Button 
              onClick={classifyWaste}
              disabled={!imageFile || isClassifying}
              className="w-full bg-gradient-primary hover:shadow-glow disabled:opacity-50"
              size="lg"
            >
              {isClassifying ? (
                <>
                  <Brain className="w-5 h-5 mr-2 animate-spin-slow" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Sparkles className="w-5 h-5 mr-2" />
                  Classify Waste
                </>
              )}
            </Button>
          </div>
        </Card>

        {/* Results Section */}
        <Card className="p-8 bg-gradient-card border-primary/20 glow-card">
          <div className="space-y-6">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-primary/20 rounded-lg flex items-center justify-center">
                <CheckCircle className="w-5 h-5 text-primary-glow" />
              </div>
              <h3 className="text-xl font-semibold">Classification Result</h3>
            </div>

            <div className="min-h-[400px] flex items-center justify-center">
              {result ? (
                <div className="text-center space-y-6 w-full">
                  <div className="text-6xl mb-4">{result.icon}</div>
                  
                  <div className="space-y-2">
                    <h4 className="text-3xl font-bold">{result.category}</h4>
                    <Badge variant="secondary" className="text-sm">
                      {result.confidence}% Confidence
                    </Badge>
                  </div>

                  <div className="bg-muted/50 rounded-lg p-6 text-left">
                    <h5 className="font-semibold mb-3 text-primary-glow">
                      ♻️ Recycling Recommendations:
                    </h5>
                    <ul className="space-y-2">
                      {result.recommendations.map((rec, index) => (
                        <li key={index} className="flex items-start gap-2 text-sm">
                          <span className="text-primary-glow">•</span>
                          <span>{rec}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              ) : (
                <div className="text-center text-muted-foreground space-y-4">
                  <AlertCircle className="w-16 h-16 mx-auto opacity-50" />
                  <div>
                    <p className="text-lg">Upload an image to see results</p>
                    <p className="text-sm">Supports JPG, PNG, and WEBP formats</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}