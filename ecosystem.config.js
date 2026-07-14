/** PM2 config — run on the server after first deploy. */
module.exports = {
  apps: [
    {
      name: "multimodal-rag",
      script: "./deploy.sh",
      interpreter: "bash",
      cwd: "/root/multimodal-rag-new",
      autorestart: true,
      max_restarts: 3,
      watch: false,
    },
  ],
};
