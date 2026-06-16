import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/aimooc/",
  plugins: [react()],
  build: {
    outDir: "../static/aimooc",
    emptyOutDir: true,
  },
});
