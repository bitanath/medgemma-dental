import React, { useState, useEffect, useRef, useCallback } from 'react';
import './App.css';

const API_BASE = '';
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
  '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FFA500', '#800080', '#b2eaa1',
  '#008000', '#000080', '#808000', '#800000', '#008080', '#C0C0C0', '#808080', '#FF69B4',
  '#4B0082', '#FFD700', '#32CD32', '#FF6347', '#40E0D0', '#EE82EE', '#F5DEB3', '#D2691E',
  '#00CED1', '#9400D3', '#FF1493', '#7FFF00', '#B8860B', '#9932CC', '#fa936d', '#20B2AA',
  '#1f1e11',
];

const TREATMENTS = ['none', 'extraction', 'restoration', 'replacement', 'rct', 'filling'];

function getBoxColor(box) {
  if (box.treatment && box.treatment.toLowerCase() !== 'none') {
    return '#FF0000';
  }
  const toothIndex = TOOTH_TYPES.indexOf(box.tooth);
  return TOOTH_COLORS[toothIndex >= 0 ? toothIndex : TOOTH_COLORS.length - 1];
}

// Lazy thumbnail component - only loads image when visible
function LazyThumbnail({ img, idx, currentIndex, onClick }) {
  const [isVisible, setIsVisible] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);
  const thumbnailRef = useRef(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          observer.disconnect();
        }
      },
      {
        rootMargin: '100px', // Start loading 100px before visible
        threshold: 0
      }
    );

    if (thumbnailRef.current) {
      observer.observe(thumbnailRef.current);
    }

    return () => observer.disconnect();
  }, []);

  return (
    <div
      ref={thumbnailRef}
      className={`thumbnail ${idx === currentIndex ? 'active' : ''}`}
      onClick={onClick}
    >
      {isVisible ? (
        <>
          {!isLoaded && <div className="thumbnail-placeholder">Loading...</div>}
          <img 
            src={`${API_BASE}/images/${img.file}`} 
            alt={img.file} 
            loading="lazy"
            onLoad={() => setIsLoaded(true)}
            style={{ opacity: isLoaded ? 1 : 0 }}
          />
        </>
      ) : (
        <div className="thumbnail-placeholder">Loading...</div>
      )}
      <div className="thumbnail-info">
        <span className="filename">{img.file}</span>
        <span className="count">{img.objectCount} boxes</span>
      </div>
    </div>
  );
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
  const [baseScale, setBaseScale] = useState(1);
  const [zoomLevel, setZoomLevel] = useState(1.56);
  const scale = baseScale * zoomLevel;
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPos, setStartPos] = useState(null);
  const [currentBox, setCurrentBox] = useState(null);
  const [selectedBox, setSelectedBox] = useState(null);
  const [boxes, setBoxes] = useState([]);
  const [dragMode, setDragMode] = useState(null); // 'move' or 'resize'
  const [dragStart, setDragStart] = useState(null);
  const [originalBox, setOriginalBox] = useState(null);
  const [hideBoxes, setHideBoxes] = useState(false);

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

  useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.key === 'Control' && e.key === 'Backspace') && selectedBox !== null) {
        e.preventDefault();
        deleteBox(selectedBox);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedBox, boxes]);

  // Auto-select first box if none selected and boxes exist
  useEffect(() => {
    if (boxes.length > 0 && selectedBox === null) {
      setSelectedBox(0);
    }
  }, [boxes, selectedBox]);

  const loadImage = useCallback((filename, index) => {
    setLoading(true);
    fetch(`${API_BASE}/api/image/${filename}`)
      .then(res => res.json())
      .then(data => {
        setCurrentImage(filename);
        setCurrentIndex(index);
        setImageData(data);
        
        // Normalize treatment values to lowercase
        const normalizedObjects = (data.objects || []).map(obj => ({
          ...obj,
          treatment: (obj.treatment || 'none').toLowerCase()
        }));
        
        // Check if only one tooth - auto create box in center
        if (normalizedObjects && normalizedObjects.length === 1) {
          const singleBox = normalizedObjects[0];
          // If box is empty (all zeros), create center box
          if (singleBox.x1 === 0 && singleBox.y1 === 0 && singleBox.x2 === 0 && singleBox.y2 === 0) {
            const centerBox = {
              ...singleBox,
              x1: 384,
              y1: 256,
              x2: 640,
              y2: 768,
              wd: 256,
              ht: 512,
              tooth: singleBox.tooth || 'unknown',
              treatment: singleBox.treatment || 'none',
              diagnosis: singleBox.diagnosis || ''
            };
            setBoxes([centerBox]);
            setSelectedBox(0);
            setHasChanges(true);
          } else {
            setBoxes(normalizedObjects);
            setSelectedBox(0);
          }
        } else {
          setBoxes(normalizedObjects);
          setSelectedBox(normalizedObjects.length > 0 ? 0 : null);
        }
        
        setHasChanges(false);
        setZoomLevel(1);
        setLoading(false);
      })
      .catch(err => {
        console.error('Failed to load image:', err);
        setLoading(false);
      });
  }, []);

  const handleZoomIn = () => {
    setZoomLevel(prev => Math.min(prev * 1.25, 1.56));
  };

  const handleZoomOut = () => {
    setZoomLevel(prev => Math.max(prev / 1.25, 0.75));
  };

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

  const getMousePos = (e) => {
    if (!canvasRef.current) return { x: 0, y: 0 };
    const rect = canvasRef.current.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) / scale,
      y: (e.clientY - rect.top) / scale
    };
  };

  const getResizeHandle = (box, x, y) => {
    const handleSize = 10 / scale;
    const handles = [
      { pos: 'nw', x: box.x1, y: box.y1 },
      { pos: 'ne', x: box.x2, y: box.y1 },
      { pos: 'sw', x: box.x1, y: box.y2 },
      { pos: 'se', x: box.x2, y: box.y2 }
    ];
    
    for (const handle of handles) {
      if (Math.abs(x - handle.x) < handleSize && Math.abs(y - handle.y) < handleSize) {
        return handle.pos;
      }
    }
    return null;
  };

  const handleMouseDown = (e) => {
    if (!canvasRef.current) return;
    const { x, y } = getMousePos(e);
    
    // Check if clicking on resize handle of selected box
    if (selectedBox !== null && boxes[selectedBox]) {
      const handle = getResizeHandle(boxes[selectedBox], x, y);
      if (handle) {
        setDragMode('resize');
        setDragStart({ x, y, handle });
        setOriginalBox({ ...boxes[selectedBox] });
        return;
      }
    }
    
    // Check if clicking inside selected box (for move)
    if (selectedBox !== null && boxes[selectedBox]) {
      const box = boxes[selectedBox];
      if (x >= box.x1 && x <= box.x2 && y >= box.y1 && y <= box.y2) {
        setDragMode('move');
        setDragStart({ x, y });
        setOriginalBox({ ...box });
        return;
      }
    }
    
    // Check if clicking on another box
    const clickedBox = boxes.findIndex((box, idx) => {
      return x >= box.x1 && x <= box.x2 && y >= box.y1 && y <= box.y2;
    });
    
    if (clickedBox !== -1) {
      setSelectedBox(clickedBox);
    } else {
      // Start drawing new box
      setIsDrawing(true);
      setStartPos({ x, y });
      setCurrentBox({ x1: x, y1: y, x2: x, y2: y });
      setSelectedBox(null);
    }
  };

  const handleMouseMove = (e) => {
    if (!canvasRef.current) return;
    const { x, y } = getMousePos(e);
    
    // Handle dragging (move or resize)
    if (dragMode && selectedBox !== null && originalBox && dragStart) {
      const dx = x - dragStart.x;
      const dy = y - dragStart.y;
      
      if (dragMode === 'move') {
        updateBox(selectedBox, {
          x1: originalBox.x1 + dx,
          y1: originalBox.y1 + dy,
          x2: originalBox.x2 + dx,
          y2: originalBox.y2 + dy
        });
      } else if (dragMode === 'resize') {
        const updates = {};
        switch (dragStart.handle) {
          case 'nw':
            updates.x1 = originalBox.x1 + dx;
            updates.y1 = originalBox.y1 + dy;
            break;
          case 'ne':
            updates.x2 = originalBox.x2 + dx;
            updates.y1 = originalBox.y1 + dy;
            break;
          case 'sw':
            updates.x1 = originalBox.x1 + dx;
            updates.y2 = originalBox.y2 + dy;
            break;
          case 'se':
            updates.x2 = originalBox.x2 + dx;
            updates.y2 = originalBox.y2 + dy;
            break;
        }
        updateBox(selectedBox, updates);
      }
      return;
    }
    
    // Handle drawing
    if (isDrawing && startPos) {
      setCurrentBox({
        x1: Math.min(startPos.x, x),
        y1: Math.min(startPos.y, y),
        x2: Math.max(startPos.x, x),
        y2: Math.max(startPos.y, y)
      });
    }
  };

  const handleMouseUp = () => {
    // End dragging
    if (dragMode) {
      setDragMode(null);
      setDragStart(null);
      setOriginalBox(null);
      return;
    }
    
    // End drawing
    if (isDrawing && currentBox) {
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
        const newBoxes = [...boxes, newBox];
        setBoxes(newBoxes);
        setSelectedBox(newBoxes.length - 1);
        setHasChanges(true);
      }
      
      setIsDrawing(false);
      setStartPos(null);
      setCurrentBox(null);
    }
  };

  const updateBox = (index, updates) => {
    const newBoxes = [...boxes];
    // Round coordinate updates to integers
    const roundedUpdates = { ...updates };
    if (updates.x1 !== undefined) roundedUpdates.x1 = Math.round(updates.x1);
    if (updates.y1 !== undefined) roundedUpdates.y1 = Math.round(updates.y1);
    if (updates.x2 !== undefined) roundedUpdates.x2 = Math.round(updates.x2);
    if (updates.y2 !== undefined) roundedUpdates.y2 = Math.round(updates.y2);
    
    newBoxes[index] = { ...newBoxes[index], ...roundedUpdates };
    if (updates.x1 !== undefined || updates.y1 !== undefined || 
        updates.x2 !== undefined || updates.y2 !== undefined) {
      const box = newBoxes[index];
      newBoxes[index].wd = Math.round(Math.abs(box.x2 - box.x1));
      newBoxes[index].ht = Math.round(Math.abs(box.y2 - box.y1));
    }
    setBoxes(newBoxes);
    setHasChanges(true);
  };

  const deleteBox = (index) => {
    const newBoxes = boxes.filter((_, i) => i !== index);
    setBoxes(newBoxes);
    setSelectedBox(newBoxes.length > 0 ? Math.min(index, newBoxes.length - 1) : null);
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
      setBaseScale(imgScale);
      
      canvas.width = 1024;
      canvas.height = 1024;
      canvas.style.width = `${1024 * scale}px`;
      canvas.style.height = `${1024 * scale}px`;
      
      ctx.drawImage(img, 0, 0);
      
      // Draw boxes only if not hidden
      if (!hideBoxes) {
        boxes.forEach((box, idx) => {
          ctx.strokeStyle = getBoxColor(box);
          ctx.lineWidth = 3;
          
          if (box.treatment && box.treatment.toLowerCase() !== 'none') {
            ctx.setLineDash([15, 5, 5, 5]);
          } else {
            ctx.setLineDash([]);
          }
          
          ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
          
          ctx.setLineDash([]);
          
          // Draw resize handles for selected box
          if (idx === selectedBox) {
            ctx.strokeStyle = '#ffffff';
            ctx.setLineDash([3, 3]);
            ctx.strokeRect(box.x1 - 3, box.y1 - 3, box.x2 - box.x1 + 6, box.y2 - box.y1 + 6);
            ctx.setLineDash([]);
            
            // Draw resize handles
            ctx.fillStyle = '#ffffff';
            const handleSize = 8;
            const handles = [
              [box.x1 - handleSize/2, box.y1 - handleSize/2],
              [box.x2 - handleSize/2, box.y1 - handleSize/2],
              [box.x1 - handleSize/2, box.y2 - handleSize/2],
              [box.x2 - handleSize/2, box.y2 - handleSize/2]
            ];
            handles.forEach(([hx, hy]) => {
              ctx.fillRect(hx, hy, handleSize, handleSize);
            });
          }
        });
      }
      
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
  }, [currentImage, boxes, selectedBox, currentBox, scale, hideBoxes]);

  if (loading && !currentImage) {
    return <div className="loading">Loading dataset...</div>;
  }

  return (
    <div className="app">
      <div className="header">
        <h1>Custom X-Ray Annotator {currentImage && <span className="filename-display">- &nbsp; &nbsp;{currentImage}</span>}</h1>
        <div className="status-bar">
          <div className="zoom-controls">
            <button onClick={handleZoomOut} className="zoom-btn">−</button>
            <span className="zoom-level">{Math.round(zoomLevel * 100)}%</span>
            <button onClick={handleZoomIn} className="zoom-btn">+</button>
          </div>
          <div className="toggle-container">
            <span className="toggle-label">Show Boxes</span>
            <button 
              className={`toggle-switch ${!hideBoxes ? 'active' : ''}`}
              onClick={() => setHideBoxes(!hideBoxes)}
            >
              <span className="toggle-slider"></span>
            </button>
          </div>
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
              <LazyThumbnail
                key={img.file}
                img={img}
                idx={idx}
                currentIndex={currentIndex}
                onClick={() => {
                  if (hasChanges) handleSave();
                  loadImage(img.file, idx);
                }}
              />
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
              style={{ 
                cursor: dragMode === 'move' ? 'move' : 
                        dragMode === 'resize' ? 'nwse-resize' : 
                        isDrawing ? 'crosshair' : 'default'
              }}
            />
          </div>
          
          <div className="properties-panel">
            {selectedBox !== null && boxes[selectedBox] ? (
              <>
                <h3>Edit Box {selectedBox + 1} / {boxes.length}</h3>
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
                  <textarea
                    value={boxes[selectedBox].diagnosis}
                    onChange={(e) => updateBox(selectedBox, { diagnosis: e.target.value })}
                    rows={6}
                  />
                </div>
                <button onClick={() => deleteBox(selectedBox)} className="delete-btn">
                  Delete Box (Del)
                </button>
              </>
            ) : (
              <>
                <h3>No Box Selected</h3>
                <p className="no-selection-text">
                  {boxes.length === 0 
                    ? "Click and drag on image to draw a box" 
                    : "Click on a box to select it"}
                </p>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
