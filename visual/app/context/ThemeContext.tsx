"use client";

import { createContext, useContext, useState, useEffect, ReactNode } from "react";

type Theme = "dark" | "light";

interface ThemeContextType {
  theme: Theme;
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>("dark");
  const [mounted, setMounted] = useState(false);

  // Initialize theme on mount
  useEffect(() => {
    // Check for saved preference or default to dark
    const savedTheme = localStorage.getItem("fairflow-theme") as Theme | null;
    if (savedTheme && (savedTheme === "dark" || savedTheme === "light")) {
      setTheme(savedTheme);
      applyTheme(savedTheme);
    } else {
      applyTheme("dark");
    }
    setMounted(true);
  }, []);

  // Apply theme when changed
  useEffect(() => {
    if (mounted) {
      applyTheme(theme);
      localStorage.setItem("fairflow-theme", theme);
    }
  }, [theme, mounted]);

  const applyTheme = (newTheme: Theme) => {
    // Apply data-theme attribute
    document.documentElement.setAttribute("data-theme", newTheme);
    
    // Also set class for Tailwind dark mode
    if (newTheme === "dark") {
      document.documentElement.classList.add("dark");
      document.documentElement.classList.remove("light");
    } else {
      document.documentElement.classList.remove("dark");
      document.documentElement.classList.add("light");
    }
  };

  const toggleTheme = () => {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
  };

  // Always render children - theme changes will apply via CSS
  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error("useTheme must be used within a ThemeProvider");
  }
  return context;
}
