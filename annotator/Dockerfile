# Multi-stage build for production

# Stage 1: Build the React frontend
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm install

# Copy source code and build
COPY . .
RUN npm run build

# Stage 2: Production server
FROM node:20-alpine

WORKDIR /app

# Copy package files and install production dependencies
COPY package*.json ./
RUN npm install --only=production

# Copy built frontend and server files
COPY --from=builder /app/dist ./dist
COPY server.js .

# Create dataset directory (will be mounted as volume)
RUN mkdir -p dataset

# Expose port
EXPOSE 3001

# Start the server
CMD ["node", "server.js"]
