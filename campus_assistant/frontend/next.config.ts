import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Required for Docker standalone deployment (copies only needed files)
  output: "standalone",
  // Allow images from any source (useful for campus photos in Phase 2)
  images: {
    remotePatterns: [],
  },
};

export default nextConfig;
