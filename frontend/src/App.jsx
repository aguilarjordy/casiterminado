import React, { useRef, useEffect, useState } from 'react'

const VOCALS = ['A','E','I','O','U']
const MAX_PER_LABEL = 100

export default function App(){ 
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const [counts, setCounts] = useState({})
  const collectRef = useRef(null)
  const [status, setStatus] = useState('Cargando...')
  const [progress, setProgress] = useState(0)
  const [prediction, setPrediction] = useState(null)

  // 🔹 para controlar frecuencia de predicciones
  const lastPredictTime = useRef(0)

  useEffect(()=>{
    const loadScripts = async ()=>{
      const s1 = document.createElement('script')
      s1.src = 'https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js'
      s1.async = true
      document.body.appendChild(s1)

      const s2 = document.createElement('script')
      s2.src = 'https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js'
      s2.async = true
      document.body.appendChild(s2)

      const s3 = document.createElement('script')
      s3.src = 'https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js'
      s3.async = true
      document.body.appendChild(s3)

      s1.onload = ()=>{
        const hands = new window.Hands({locateFile: (file)=>`https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`})
        hands.setOptions({maxNumHands:1, modelComplexity:1, minDetectionConfidence:0.7, minTrackingConfidence:0.7})
        hands.onResults(onResults)

        if(videoRef.current){
          const camera = new window.Camera(videoRef.current, {
            onFrame: async ()=>{ await hands.send({image: videoRef.current}) },
            width: 640, height: 480
          })
          camera.start()
          setStatus('Listo - coloca la mano frente a la cámara')
        }
      }
    }
    loadScripts()
    fetchCounts()
  },[])

  const onResults = (results)=>{
    const canvas = canvasRef.current; const ctx = canvas.getContext('2d')
    const W = canvas.width = results.image.width; const H = canvas.height = results.image.height
    ctx.clearRect(0,0,W,H)
    try{ ctx.drawImage(results.image,0,0,W,H) }catch(e){}

    if(results.multiHandLandmarks && results.multiHandLandmarks.length>0){
      const landmarks = results.multiHandLandmarks[0]
      if(window.drawConnectors && window.drawLandmarks){
        window.drawConnectors(ctx, landmarks, window.HAND_CONNECTIONS, {color:'#06b6d4', lineWidth:2})
        window.drawLandmarks(ctx, landmarks, {color:'#06b6d4', lineWidth:1})
      }
      const scaled = landmarks.map(p=>[p.x * W, p.y * H, p.z || 0])
      window.currentLandmarks = scaled

      // 🔹 Auto-predict cada 600ms
      const now = Date.now()
      if (now - lastPredictTime.current > 600) {
        lastPredictTime.current = now
        autoPredict(scaled)
      }

      if(collectRef.current && collectRef.current.active && collectRef.current.label){
        fetch('http://127.0.0.1:5000/upload_landmarks',{
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body:JSON.stringify({label:collectRef.current.label,landmarks:scaled})
        })
        collectRef.current.count = (collectRef.current.count||0)+1
        setProgress(collectRef.current.count)
      }
    } else {
      window.currentLandmarks = null
    }
  }

  async function autoPredict(landmarks){
    try{
      const res = await fetch('http://127.0.0.1:5000/predict_landmarks',{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({landmarks})
      })
      const j = await res.json()
      if(res.ok){ 
        setPrediction(j.prediction + ' (' + (j.confidence*100).toFixed(1) + '%)') 
      }
    }catch(e){ 
      setStatus('Error: '+e.message) 
    }
  }

  async function fetchCounts(){
    try{ const res = await fetch('http://127.0.0.1:5000/count'); const j = await res.json(); setCounts(j||{}) }catch(e){}
  }

  const startCollect = (label)=>{
    if(collectRef.current && collectRef.current.active) return
    collectRef.current = {active:true, label, count:0}
    setStatus('Recolectando '+label)
    setProgress(0)
  }

  const stopCollect = ()=>{
    if(collectRef.current){ collectRef.current.active=false; collectRef.current=null }
    setStatus('Detenido'); setTimeout(fetchCounts,300); setProgress(0)
  }

  const handleTrain = async ()=>{
    setStatus('Entrenando...')
    try{
      const res = await fetch('http://127.0.0.1:5000/train_landmarks',{method:'POST'})
      const j = await res.json()
      if(res.ok) setStatus('Entrenado correctamente')
      else setStatus('Error: '+(j.error||'Error en entrenamiento'))
    }catch(e){
      setStatus('Error: '+e.message)
    }
  }

  return (
    <div className="container">
      <div className="left">
        <div className="card-title">Reconocimiento de Señas</div>
        <div className="video-wrap">
          <video ref={videoRef} autoPlay playsInline muted></video>
          <canvas ref={canvasRef} className="overlay-canvas"></canvas>
        </div>
        <div className="controls">
          <button className="button" onClick={fetchCounts}>Actualizar</button>
          <button className="button" onClick={handleTrain}>Entrenar</button>
          <button className="button red" onClick={stopCollect}>Detener</button>
        </div>
        <div className="small">Estado: {status} {progress>0 && `- ${progress}/${MAX_PER_LABEL}`}</div>
        <div className="prediction-box">{prediction || '-'}</div>
      </div>

      <div className="right">
        <div className="card-title">Recolección</div>
        <div className="small">Recolecta hasta {MAX_PER_LABEL} muestras por clase</div>
        {VOCALS.map(v=>{
          const current = counts[v]||0; 
          const pct = Math.round((current / MAX_PER_LABEL)*100)
          return (
            <div key={v} style={{marginTop:12}}>
              <div className="label-row"><div><strong>{v}</strong></div><div className="small">{current}/{MAX_PER_LABEL}</div></div>
              <div className="progress"><div className="progress-inner" style={{width:`${pct}%`}}></div></div>
              <div style={{display:'flex', gap:8, marginTop:8}}>
                <button className="button" onClick={()=>startCollect(v)} disabled={current>=MAX_PER_LABEL}>Recolectar {v}</button>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
