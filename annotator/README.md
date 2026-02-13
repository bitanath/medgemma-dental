# Dental X-Ray Annotator

## Quick Start

### Development
```bash
# Install dependencies
npm install

# Start both backend and frontend
npm start
```

Or run separately:
```bash
# Terminal 1 - Backend
npm run server

# Terminal 2 - Frontend
npm run dev
```

### Production Build
```bash
# Build frontend and start production server
npm run start:prod
```

## Docker Deployment

### Build and Run with Docker

```bash
# Build the image
docker build -t annotator .

# Run with dataset volume mounted
docker run -p 3001:3001 -v $(pwd)/../dataset:/app/dataset annotator
```

### Docker Compose (Recommended)

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Environment Variables

- `PORT`: Server port (default: 3001)

### Volume Mounting

The dataset folder should be mounted at `/app/dataset` inside the container:
- `dataset.json` - Annotation data
- Image files (jpg/png)

## Features

- 508 images support with thumbnail gallery
- 32 distinct colors for each tooth type
- Red dashed boxes for treatments
- White handles on selected boxes (drag to move/resize)
- Auto-save every 30 seconds + manual save
- Rolling backups (max 3)
- Zoom in/out (75% - 156%)
- Toggle to hide/show boxes
- Delete key support (Backspace on Mac)
- Case-insensitive treatment handling
- Large textarea for diagnosis
