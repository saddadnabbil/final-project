/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  // Improve hot reload on Windows
  webpack: (config, { dev }) => {
    if (dev) {
      // Improve file watching on Windows
      config.watchOptions = {
        poll: 500, // Check for changes every 500ms
        aggregateTimeout: 200, // Delay rebuild after change
        ignored: ['**/node_modules', '**/.next', '**/.git'],
      }
    }
    return config
  },
  // Development optimizations
  ...(process.env.NODE_ENV === 'development' && {
    // Disable static optimization in development for better hot reload
    output: undefined,
    // Enable React Fast Refresh
    reactStrictMode: true,
  }),
}

module.exports = nextConfig
