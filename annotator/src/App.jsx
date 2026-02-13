import React, { useState, useEffect, useRef, useCallback } from 'react';
import './App.css';

const API_BASE = 'http://localhost:3001';
const AUTOSAVE_INTERVAL = 30000;

const TOOTH_TYPES = [
  'upper_right_central_incisor', 'upper_right_lateral_incisor', 'upper_right_canine',
  'upper_right_first_premolar', 'upper_right_second_premolar', 'upper_right_first_molar',
  'upper_right_second_molar', 'upper_right_third_molar',
  'upper_left_central_incisor', 'upper_left_lateral_incisor', 'upper_left_canine',
  'upper_left_first_premolar', 'upper_left_second_premolar', 'upper_left_first_molar',
  'upper_left_second_molar', 'upper_left_third_molar',
  'lower_right_central_incisor', 'lower_right_lateral_incisor', 'lower_right_canine',
  'lower_right_first_premolar', 'lower_right_second_premolar', 'lower_right_first_molar',
  'lower_right_second_molar', 'lower_right_third_molar',
  'lower_left_central_incisor', 'lower_left_lateral_incisor', 'lower_left_canine',
  'lower_left_first_premolar', 'lower_left_second_premolar', 'lower_left_first_molar',
  'lower_left_second_molar', 'lower_left_third_molar',
  'unknown'
];

const TOOTH_COLORS = [
  '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FFA500', '#800080',
  '#008000', '#000080', '#808000', '#800000', '#008080', '#C0C0C0', '#808080', '#FF69B4',
  '#4B0082', '#FFD700', '#32CD32', '#FF6347', '#40E0D0', '#EE82EE', '#F5DEB3', '#D2691E',
  '#DC143C', '#00CED1', '#9400D3', '#FF1493', '#7FFF00', '#B8860B', '#9932CC', '#FF4500'
];

const TREATMENTS = ['none', 'extraction', 'restoration', 'rct', 'filling'];

function getBoxColor(box) {
  if (box.treatment && box.treatment !== 'none') {
    return '#FF0000';
  }
  const toothIndex = TOOTH_TYPES.indexOf(box.tooth);
  return TOOTH_COLORS[toothIndex >= 0 ? toothIndex : TOOTH_COLORS.length - 1];
}

function App() {
  const [images, setImages] = useState([]);
  const [currentImage, setCurrentImage] = useState(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [imageData, setImageData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [lastSaved, setLastSaved] = useState(null);
  const [hasChanges, setHasChanges] = useState(false);
  
  const canvasRef = useRef(null);
  const [scale, setScale] = useState(1);
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPos, setStartPos] = useState(null);
  const [currentBox, setCurrentBox] = useState(null);
  const [selectedBox, setSelectedBox] = useState(null);
  const [boxes, setBoxes] = useState([]);

  useEffect(() => {
    fetch(`${API_BASE}/api/dataset`)
      .then(res => res.json())
      .then(data => {
        setImages(data.images);
        if (data.images.length > 0) {
          loadImage(data.images[0].file, 0);
        }
        setLoading(false);
      })
      .catch(err => {
        console.error('Failed to load dataset:', err);
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    if (!hasChanges || !currentImage) return;
    const interval = setInterval(() => handleSave(), AUTOSAVE_INTERVAL);
    return () => clearInterval(interval);
  }, [hasChanges, currentImage, boxes]);

  const loadImage = useCallback((filename, index) => {
    setLoading(true);
    fetch(`${API_BASE}/api/image/${filename}`)
      .then(res => res.json())
      .then(data => {
        setCurrentImage(filename);
        setCurrentIndex(index);
        setImageData(data);
        setBoxes(data.objects || []);
        setSelectedBox(null);
        setHasChanges(false);
        setLoading(false);
      })
      .catch(err => {
        console.error('Failed to load image:', err);
        setLoading(false);
      });
  }, []);

  const handleSave = useCallback(async () => {
    if (!currentImage) return;
    setSaving(true);
    try {
      const response = await fetch(`${API_BASE}/api/save/${currentImage}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ objects: boxes })
      });
      if (response.ok) {
        setLastSaved(new Date());
        setHasChanges(false);
      }
    } catch (err) {
      console.error('Save error:', err);
    }
    setSaving(false);
  }, [currentImage, boxes]);

  const handleMouseDown = (e) => {
    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) / scale;
    const y = (e.clientY - rect.top) / scale;
    
    const clickedBox = boxes.findIndex((box, idx) => {
      return x >= box.x1 && x <= box.x2 && y >= box.y1 && y <= box.y2;
    });
    
    if (clickedBox !== -1) {
      setSelectedBox(clickedBox);
    } else {
      setIsDrawing(true);
      setStartPos({ x, y });
      setCurrentBox({ x1: x, y1: y, x2: x, y2: y });
      setSelectedBox(null);
    }
  };

  const handleMouseMove = (e) => {
    if (!isDrawing || !startPos || !canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) / scale;
    const y = (e.clientY - rect.top) / scale;
    
    setCurrentBox({
      x1: Math.min(startPos.x, x),
      y1: Math.min(startPos.y, y),
      x2: Math.max(startPos.x, x),
      y2: Math.max(startPos.y, y)
    });
  };

  const handleMouseUp = () => {
    if (!isDrawing || !currentBox) return;
    
    const width = currentBox.x2 - currentBox.x1;
    const height = currentBox.y2 - currentBox.y1;
    
    if (width > 10 && height > 10) {
      const newBox = {
        object_id: `item-${Date.now()}_unknown_${boxes.length}`,
        x1: Math.round(currentBox.x1),
        y1: Math.round(currentBox.y1),
        x2: Math.round(currentBox.x2),
        y2: Math.round(currentBox.y2),
        wd: Math.round(width),
        ht: Math.round(height),
        tooth: 'unknown',
        treatment: 'none',
        diagnosis: ''
      };
      setBoxes([...boxes, newBox]);
      setSelectedBox(boxes.length);
      setHasChanges(true);
    }
    
    setIsDrawing(false);
    setStartPos(null);
    setCurrentBox(null);
  };

  const updateBox = (index, updates) => {
    const newBoxes = [...boxes];
    newBoxes[index] = { ...newBoxes[index], ...updates };
    if (updates.x1 !== undefined || updates.y1 !== undefined || 
        updates.x2 !== undefined || updates.y2 !== undefined) {
      const box = newBoxes[index];
      newBoxes[index].wd = box.x2 - box.x1;
      newBoxes[index].ht = box.y2 - box.y1;
    }
    setBoxes(newBoxes);
    setHasChanges(true);
  };

  const deleteBox = (index) => {
    setBoxes(boxes.filter((_, i) => i !== index));
    setSelectedBox(null);
    setHasChanges(true);
  };

  useEffect(() => {
    if (!canvasRef.current || !currentImage) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const img = new Image();
    
    img.onload = () => {
      const containerWidth = canvas.parentElement.clientWidth - 40;
      const containerHeight = canvas.parentElement.clientHeight - 40;
      const imgScale = Math.min(containerWidth / 1024, containerHeight / 1024, 1);
      setScale(imgScale);
      
      canvas.width = 1024;
      canvas.height = 1024;
      canvas.style.width = `${1024 * imgScale}px`;
      canvas.style.height = `${1024 * imgScale}px`;
      
      ctx.drawImage(img, 0, 0);
      
      boxes.forEach((box, idx) => {
        ctx.strokeStyle = getBoxColor(box);
        ctx.lineWidth = 3;
        
        if (box.treatment && box.treatment !== 'none') {
          ctx.setLineDash([15, 5, 5, 5]);
        } else {
          ctx.setLineDash([]);
        }
        
        ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
        
        ctx.setLineDash([]);
        
        if (idx === selectedBox) {
          ctx.strokeStyle = '#ffffff';
          ctx.setLineDash([3, 3]);
          ctx.strokeRect(box.x1 - 3, box.y1 - 3, box.x2 - box.x1 + 6, box.y2 - box.y1 + 6);
          ctx.setLineDash([]);
        }
      });
      
      if (currentBox) {
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.strokeRect(
          currentBox.x1, 
          currentBox.y1, 
          currentBox.x2 - currentBox.x1, 
          currentBox.y2 - currentBox.y1
        );
        ctx.setLineDash([]);
      }
    };
    
    img.src = `${API_BASE}/images/${currentImage}`;
  }, [currentImage, boxes, selectedBox, currentBox, scale]);

  if (loading && !currentImage) {
    return <div className="loading">Loading dataset...</div>;
  }

  return (
    <div className="app">
      <div className="header">
        <h1>Dental X-Ray Annotator</h1>
        <div className="status-bar">
          {hasChanges && <span className="unsaved">● Unsaved changes</span>}
          {saving && <span className="saving">Saving...</span>}
          {lastSaved && !saving && (
            <span className="saved">Last saved: {lastSaved.toLocaleTimeString()}</span>
          )}
          <button onClick={handleSave} disabled={saving || !hasChanges} className="save-btn">
            Save Now
          </button>
        </div>
      </div>

      <div className="main-content">
        <div className="sidebar">
          <div className="nav-controls">
            <button onClick={() => {
              if (currentIndex > 0) {
                if (hasChanges) handleSave();
                loadImage(images[currentIndex - 1].file, currentIndex - 1);
              }
            }} disabled={currentIndex === 0}>← Prev</button>
            <span className="counter">{currentIndex + 1} / {images.length}</span>
            <button onClick={() => {
              if (currentIndex < images.length - 1) {
                if (hasChanges) handleSave();
                loadImage(images[currentIndex + 1].file, currentIndex + 1);
              }
            }} disabled={currentIndex === images.length - 1}>Next →</button>
          </div>
          
          <div className="thumbnail-list">
            {images.map((img, idx) => (
              <div
                key={img.file}
                className={`thumbnail ${idx === currentIndex ? 'active' : ''}`}
                onClick={() => {
                  if (hasChanges) handleSave();
                  loadImage(img.file, idx);
                }}
              >
                <img src={`${API_BASE}/images/${img.file}`} alt={img.file} loading="lazy" />
                <div className="thumbnail-info">
                  <span className="filename">{img.file}</span>
                  <span className="count">{img.objectCount} boxes</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="workspace">
          <div className="canvas-container">
            <canvas
              ref={canvasRef}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
              style={{ cursor: isDrawing ? 'crosshair' : 'default' }}
            />
          </div>
          
          {selectedBox !== null && boxes[selectedBox] && (
            <div className="properties-panel">
              <h3>Edit Box</h3>
              <div className="form-group">
                <label>Tooth Type:</label>
                <select 
                  value={boxes[selectedBox].tooth}
                  onChange={(e) => updateBox(selectedBox, { tooth: e.target.value })}
                >
                  {TOOTH_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
                </select>
              </div>
              <div className="form-group">
                <label>Treatment:</label>
                <select 
                  value={boxes[selectedBox].treatment}
                  onChange={(e) => updateBox(selectedBox, { treatment: e.target.value })}
                >
                  {TREATMENTS.map(t => <option key={t} value={t}>{t}</option>)}
                </select>
              </div>
              <div className="form-group">
                <label>Diagnosis:</label>
                <input 
                  type="text"
                  value={boxes[selectedBox].diagnosis}
                  onChange={(e) => updateBox(selectedBox, { diagnosis: e.target.value })}
                />
              </div>
              <button onClick={() => deleteBox(selectedBox)} className="delete-btn">
                Delete Box
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
