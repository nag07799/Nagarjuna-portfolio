import React from "react";
import { useTheme } from "../../context/ThemeContext";

const ThemeToggle = ({ className = "" }) => {
  const { theme, themes, setTheme } = useTheme();

  return (
    <div className={`flex flex-col gap-2 ${className}`}>
      <span className="text-xs uppercase tracking-[0.2em] text-gray-400">
        Theme
      </span>
      <div className="flex items-center gap-2">
        {themes.map((option) => {
          const isActive = option.id === theme;
          return (
            <button
              type="button"
              key={option.id}
              onClick={() => setTheme(option.id)}
              className={`relative w-9 h-9 rounded-full border border-gray-700 transition duration-200 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-designColor ${
                isActive
                  ? "ring-2 ring-offset-2 ring-designColor ring-offset-[var(--color-body)]"
                  : "opacity-70 hover:opacity-100"
              }`}
            >
              <span className="sr-only">{`Switch to ${option.label} theme`}</span>
              <span
                aria-hidden="true"
                className="absolute inset-0 rounded-full"
                style={{ background: option.preview }}
              />
            </button>
          );
        })}
      </div>
    </div>
  );
};

export default ThemeToggle;
