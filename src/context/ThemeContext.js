import React, {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";

const THEME_STORAGE_KEY = "portfolio-theme";

const themeOptions = [
  {
    id: "sunset",
    label: "Sunset",
    preview: "linear-gradient(135deg, #3b1d28, #f97316)",
  },
  {
    id: "dark",
    label: "Dark",
    preview: "linear-gradient(135deg, #1e2024, #ff014f)",
  },
  {
    id: "light",
    label: "Light",
    preview: "linear-gradient(135deg, #e0f2fe, #2563eb)",
  },
];

const ThemeContext = createContext({
  theme: themeOptions[0].id,
  themes: themeOptions,
  setTheme: () => {},
});

const getInitialTheme = () => {
  if (typeof window === "undefined") {
    return themeOptions[0].id;
  }
  return (
    window.localStorage.getItem(THEME_STORAGE_KEY) ?? themeOptions[0].id
  );
};

export const ThemeProvider = ({ children }) => {
  const [theme, setTheme] = useState(getInitialTheme);

  useEffect(() => {
    if (typeof document !== "undefined") {
      document.documentElement.setAttribute("data-theme", theme);
    }
    if (typeof window !== "undefined") {
      window.localStorage.setItem(THEME_STORAGE_KEY, theme);
    }
  }, [theme]);

  const value = useMemo(
    () => ({
      theme,
      themes: themeOptions,
      setTheme,
    }),
    [theme]
  );

  return (
    <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
  );
};

export const useTheme = () => useContext(ThemeContext);
