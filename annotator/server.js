import express from 'express';
import cors from 'cors';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

// Check if dist folder exists (production build)
const distPath = path.join(__dirname, 'dist');
const isProduction = fs.existsSync(distPath);

// Dataset directory - check both locations (parent for dev, local for Docker)
let DATASET_DIR = path.join(__dirname, '..', 'dataset_all');
if (!fs.existsSync(DATASET_DIR)) {
  // Try local dataset folder (for Docker)
  DATASET_DIR = path.join(__dirname, 'dataset_all');
}

const DATASET_JSON = path.join(DATASET_DIR, 'dataset.json');

function getBackupPath(index) {
  return path.join(DATASET_DIR, `dataset.json.backup.${index}`);
}

function rotateBackups() {
  try {
    if (fs.existsSync(getBackupPath(3))) {
      fs.unlinkSync(getBackupPath(3));
    }
    if (fs.existsSync(getBackupPath(2))) {
      fs.renameSync(getBackupPath(2), getBackupPath(3));
    }
    if (fs.existsSync(getBackupPath(1))) {
      fs.renameSync(getBackupPath(1), getBackupPath(2));
    }
    if (fs.existsSync(DATASET_JSON)) {
      fs.copyFileSync(DATASET_JSON, getBackupPath(1));
    }
    return true;
  } catch (error) {
    console.error('Backup rotation failed:', error);
    return false;
  }
}

app.get('/api/dataset', (req, res) => {
  try {
    const data = JSON.parse(fs.readFileSync(DATASET_JSON, 'utf8'));
    const images = data.map(item => ({
      file: item.file,
      objectCount: item.objects ? item.objects.length : 0
    }));
    res.json({ images, total: images.length });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/image/:filename', (req, res) => {
  try {
    const data = JSON.parse(fs.readFileSync(DATASET_JSON, 'utf8'));
    const imageData = data.find(item => item.file === req.params.filename);
    if (!imageData) {
      return res.status(404).json({ error: 'Image not found' });
    }
    
    res.json(imageData);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/save/:filename', (req, res) => {
  try {
    if (!rotateBackups()) {
      return res.status(500).json({ error: 'Backup rotation failed' });
    }
    
    const data = JSON.parse(fs.readFileSync(DATASET_JSON, 'utf8'));
    const index = data.findIndex(item => item.file === req.params.filename);
    
    if (index === -1) {
      return res.status(404).json({ error: 'Image not found' });
    }
    
    data[index].objects = req.body.objects;
    fs.writeFileSync(DATASET_JSON, JSON.stringify(data, null, 2));
    res.json({ success: true, message: 'Saved successfully' });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Serve images from dataset
app.use('/images', express.static(DATASET_DIR));

// Serve static files in production
if (isProduction) {
  app.use(express.static(distPath));
  
  // Handle client-side routing
  app.get('*', (req, res) => {
    res.sendFile(path.join(distPath, 'index.html'));
  });
}

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
  if (isProduction) {
    console.log('Production mode: Serving static files from dist/');
  } else {
    console.log('Development mode: Run npm run dev for frontend');
  }
});
