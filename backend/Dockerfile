FROM node:18-alpine

WORKDIR /app

# Install dependencies FIRST
COPY backend/package*.json ./
RUN npm install

# Copy Code
COPY backend/ .

# Expose Port
EXPOSE 3000

# Run
CMD ["node", "server.js"]
