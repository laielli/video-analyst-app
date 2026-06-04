/** @type {import('next').NextConfig} */
const nextConfig = {
  // Static export -> Azure Static Web Apps (avoids SSR constraints). The page is a
  // client component that talks to the FastAPI/SSE backend, so export is fine.
  output: "export",
  images: { unoptimized: true },
  reactStrictMode: true,
};

export default nextConfig;
