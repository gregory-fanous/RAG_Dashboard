/** @type {import('next').NextConfig} */
const backendApiBase =
  (process.env.BACKEND_API_BASE ?? process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000").replace(
    /\/$/,
    "",
  );

const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${backendApiBase}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
