import { useState, useRef, useCallback, useEffect } from 'react'

// ─── Types ────────────────────────────────────────────────────────────────────

type Page = 'verification' | 'history' | 'system'
type PipelineState = 'idle' | 'loading' | 'success' | 'failure' | 'search_error' | 'error'

interface LogEntry {
  time: string
  message: string
  status: 'info' | 'success' | 'warning' | 'error'
}

interface PipelineStep {
  id: number
  label: string
  sublabel: string
  state: 'pending' | 'active' | 'done' | 'failed' | 'skipped'
}

interface ApiResultData {
  success: boolean
  face_detected: boolean
  detection_score?: number
  bbox?: number[]
  embedding_generated?: boolean
  search_status?: 'success' | 'failed'
  error_stage?: string
  filename?: string
  candidates_evaluated: number
  match_found: boolean
  highest_similarity?: number
  threshold: number
  error?: string
  candidate?: {
    title: string
    source_url: string
    domain: string
    candidate_image_url?: string
    similarity: number
  } | null
  blockchain?: {
    fingerprint: string
    record_id: string
    block_hash?: string
    previous_hash?: string
    chain_integrity: string
    reverification: string
    confidence: number
  } | null
  logs?: LogEntry[]
}

interface HistoryRecord {
  id: string
  date: string
  input: string
  candidates: number
  similarity: string
  source: string
  blockchain: string
  status: 'success' | 'failure'
}

interface SystemHealth {
  status: string
  system: string
  face_engine: string
  embedding_dimensions: number
  default_threshold: number
  serpapi_configured: boolean
  blockchain: string
  timestamp: string
}

// ─── Constants ────────────────────────────────────────────────────────────────

const PIPELINE_STEPS_INITIAL: PipelineStep[] = [
  { id: 1, label: 'Face Detection', sublabel: 'InsightFace', state: 'pending' },
  { id: 2, label: 'Search', sublabel: 'Google Lens', state: 'pending' },
  { id: 3, label: 'Verify', sublabel: 'Cosine Match', state: 'pending' },
  { id: 4, label: 'Blockchain', sublabel: 'SHA-256 Ledger', state: 'pending' },
  { id: 5, label: 'Re-Verify', sublabel: 'On-Chain Audit', state: 'pending' },
]

// ─── Utility Functions ────────────────────────────────────────────────────────

function cn(...classes: (string | false | undefined | null)[]) {
  return classes.filter(Boolean).join(' ')
}

function formatNow() {
  return new Date().toLocaleTimeString('en-GB', { hour12: false })
}

function truncateHash(hash?: string, start = 10, end = 8): string {
  if (!hash || hash === '--') return '--'
  if (hash.length <= start + end + 3) return hash
  return `${hash.slice(0, start)}...${hash.slice(-end)}`
}

function truncateFilename(name: string, maxLen = 30): string {
  if (!name) return ''
  if (name.length <= maxLen) return name
  const lastDot = name.lastIndexOf('.')
  const ext = lastDot !== -1 ? name.slice(lastDot) : ''
  const base = lastDot !== -1 ? name.slice(0, lastDot) : name
  const keepLen = maxLen - ext.length - 3
  if (keepLen <= 0) return name.slice(0, maxLen - 3) + '...'
  return `${base.slice(0, keepLen)}...${ext}`
}

// ─── Icons ────────────────────────────────────────────────────────────────────

function IconFaceVerify() {
  return (
    <svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="28" height="28" rx="7" fill="#111827" />
      <circle cx="14" cy="12" r="5.5" stroke="white" strokeWidth="1.5" fill="none" />
      <path d="M11 12c0-1.657 1.343-3 3-3" stroke="white" strokeWidth="1.5" strokeLinecap="round" />
      <path d="M8.5 20.5c0-3.038 2.462-5.5 5.5-5.5s5.5 2.462 5.5 5.5" stroke="white" strokeWidth="1.5" strokeLinecap="round" />
      <circle cx="21" cy="8" r="4" fill="#059669" />
      <path d="M19 8l1.5 1.5 2.5-2.5" stroke="white" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

function IconUpload() {
  return (
    <svg width="36" height="36" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M20 26V14M20 14l-5 5M20 14l5 5" stroke="#9CA3AF" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M12 30h16M8 22c0 4.418 3.582 8 8 8h8c4.418 0 8-3.582 8-8" stroke="#9CA3AF" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  )
}

function IconShield() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M8 1.5L2.5 4v4c0 3.038 2.364 5.88 5.5 6.5 3.136-.62 5.5-3.462 5.5-6.5V4L8 1.5z" stroke="#059669" strokeWidth="1.2" fill="none" />
      <path d="M5.5 8l1.8 1.8L10.5 6.5" stroke="#059669" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

function IconCopy({ textToCopy, label }: { textToCopy: string; label?: string }) {
  const [copied, setCopied] = useState(false)

  const handleClick = async () => {
    if (!textToCopy || textToCopy === '--') return

    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(textToCopy)
      } else {
        const textArea = document.createElement('textarea')
        textArea.value = textToCopy
        textArea.style.position = 'fixed'
        textArea.style.left = '-999999px'
        textArea.style.top = '-999999px'
        document.body.appendChild(textArea)
        textArea.focus()
        textArea.select()
        document.execCommand('copy')
        textArea.remove()
      }
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch (err) {
      console.error('Failed to copy text:', err)
    }
  }

  return (
    <button
      onClick={handleClick}
      className="inline-flex items-center gap-1 px-2 py-1 text-[11px] font-medium text-gray-600 hover:text-gray-900 bg-gray-100 hover:bg-gray-200 rounded-md transition-colors flex-shrink-0"
      title={`Copy full ${label || 'value'} (64-character SHA-256) to clipboard`}
    >
      {copied ? (
        <span className="text-emerald-600 font-semibold flex items-center gap-1">
          <svg width="12" height="12" viewBox="0 0 14 14" fill="none"><path d="M2 7l3.5 3.5 6.5-7" stroke="#059669" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg>
          Copied
        </span>
      ) : (
        <span className="flex items-center gap-1">
          <svg width="12" height="12" viewBox="0 0 14 14" fill="none"><rect x="4.5" y="4.5" width="8" height="8" rx="1.5" stroke="currentColor" strokeWidth="1.2" /><path d="M4.5 9.5H3a1.5 1.5 0 01-1.5-1.5V3A1.5 1.5 0 013 1.5h5A1.5 1.5 0 019.5 3v1.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" /></svg>
          Copy
        </span>
      )}
    </button>
  )
}

function IconSpinner() {
  return (
    <svg className="animate-spin" width="16" height="16" viewBox="0 0 20 20" fill="none">
      <circle cx="10" cy="10" r="8" stroke="#E5E7EB" strokeWidth="2.5" />
      <path d="M10 2a8 8 0 018 8" stroke="#2563EB" strokeWidth="2.5" strokeLinecap="round" />
    </svg>
  )
}

// ─── Shared UI Primitives ─────────────────────────────────────────────────────

function Button({ children, variant = 'primary', onClick, className = '', disabled = false }: {
  children: React.ReactNode
  variant?: 'primary' | 'outline' | 'ghost'
  onClick?: () => void
  className?: string
  disabled?: boolean
}) {
  const base = 'inline-flex items-center justify-center gap-2 text-xs font-semibold rounded-lg transition-all duration-150 cursor-pointer select-none disabled:opacity-50 disabled:cursor-not-allowed'
  const variants = {
    primary: 'bg-[#111827] text-white px-4 py-2 hover:bg-[#1f2937] active:bg-[#0f172a] shadow-sm',
    outline: 'border border-[#E5E7EB] text-[#111827] px-4 py-2 hover:bg-gray-50 active:bg-gray-100',
    ghost: 'text-[#667085] px-3 py-1.5 hover:bg-gray-100 hover:text-[#111827]',
  }
  return (
    <button className={cn(base, variants[variant], className)} onClick={onClick} disabled={disabled}>
      {children}
    </button>
  )
}

function StatusPill({ status, label }: { status: 'success' | 'info' | 'warning' | 'error' | 'neutral'; label: string }) {
  const styles = {
    success: 'bg-emerald-50 text-emerald-700 border-emerald-200',
    info: 'bg-blue-50 text-blue-700 border-blue-200',
    warning: 'bg-amber-50 text-amber-700 border-amber-200',
    error: 'bg-red-50 text-red-700 border-red-200',
    neutral: 'bg-gray-100 text-gray-600 border-gray-200',
  }
  const dots = {
    success: 'bg-emerald-500',
    info: 'bg-blue-500',
    warning: 'bg-amber-500',
    error: 'bg-red-500',
    neutral: 'bg-gray-400',
  }
  return (
    <span className={cn('inline-flex items-center gap-1.5 text-[11px] font-semibold px-2.5 py-0.5 rounded-full border', styles[status])}>
      <span className={cn('w-1.5 h-1.5 rounded-full', dots[status])} />
      {label}
    </span>
  )
}

function Card({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={cn('bg-white border border-[#E5E7EB] rounded-xl shadow-sm', className)}>
      {children}
    </div>
  )
}

function MetricCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <Card className="p-3.5 flex flex-col justify-between">
      <p className="text-[10px] font-semibold text-[#98A2B3] uppercase tracking-wider mb-1">{label}</p>
      <div className="flex items-baseline justify-between">
        <p className="text-xl font-bold text-[#111827] font-mono leading-none">{value}</p>
        {sub && <span className="text-[10px] text-[#98A2B3] font-medium">{sub}</span>}
      </div>
    </Card>
  )
}

// ─── Navbar ───────────────────────────────────────────────────────────────────

function Navbar({ page, setPage, isOnline }: { page: Page; setPage: (p: Page) => void; isOnline: boolean | null }) {
  const navItems: { id: Page; label: string }[] = [
    { id: 'verification', label: 'Verification' },
    { id: 'history', label: 'History' },
    { id: 'system', label: 'System' },
  ]

  return (
    <header className="bg-white border-b border-[#E5E7EB] sticky top-0 z-50">
      <div className="max-w-4xl mx-auto px-4 h-14 flex items-center justify-between gap-6">
        <div className="flex items-center gap-3 flex-shrink-0">
          <IconFaceVerify />
          <div className="flex items-center gap-2">
            <span className="font-bold text-[#111827] text-base tracking-tight">FaceVerify</span>
            <span className="text-[10px] font-medium text-[#667085] bg-gray-100 border border-[#E5E7EB] px-2 py-0.5 rounded-full">
              HH Goa 2026 · Task 3
            </span>
          </div>
        </div>

        <nav className="flex items-center gap-1">
          {navItems.map(item => (
            <button
              key={item.id}
              onClick={() => setPage(item.id)}
              className={cn(
                'px-3 py-1.5 text-xs rounded-lg transition-all duration-150 font-semibold',
                page === item.id
                  ? 'bg-[#111827] text-white'
                  : 'text-[#667085] hover:text-[#111827] hover:bg-gray-50'
              )}
            >
              {item.label}
            </button>
          ))}
        </nav>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 text-xs text-[#667085]">
            <span className={cn(
              'w-2 h-2 rounded-full',
              isOnline === true ? 'bg-emerald-500 animate-pulse' : isOnline === false ? 'bg-red-500' : 'bg-gray-400'
            )} />
            <span className="hidden sm:inline">{isOnline === true ? 'System Ready' : isOnline === false ? 'Backend Offline' : 'Checking...'}</span>
          </div>
        </div>
      </div>
    </header>
  )
}

// ─── Region Adjuster Component ────────────────────────────────────────────────

function RegionAdjuster({
  imageUrl,
  initialBbox,
  onConfirm,
  onCancel,
}: {
  imageUrl: string
  initialBbox: number[] | null
  onConfirm: (box: number[]) => void
  onCancel: () => void
}) {
  const imgRef = useRef<HTMLImageElement>(null)
  const [naturalSize, setNaturalSize] = useState<{ w: number; h: number } | null>(null)
  const [box, setBox] = useState<{ x: number; y: number; w: number; h: number }>({ x: 20, y: 15, w: 60, h: 70 })

  const handleImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const nw = e.currentTarget.naturalWidth
    const nh = e.currentTarget.naturalHeight
    setNaturalSize({ w: nw, h: nh })

    if (initialBbox && initialBbox.length === 4 && nw > 0 && nh > 0) {
      const [xmin, ymin, xmax, ymax] = initialBbox
      const px = Math.max(0, Math.min(100, (xmin / nw) * 100))
      const py = Math.max(0, Math.min(100, (ymin / nh) * 100))
      const pw = Math.max(5, Math.min(100 - px, ((xmax - xmin) / nw) * 100))
      const ph = Math.max(5, Math.min(100 - py, ((ymax - ymin) / nh) * 100))
      setBox({ x: px, y: py, w: pw, h: ph })
    }
  }

  const [dragState, setDragState] = useState<{
    mode: 'move' | 'tl' | 'tr' | 'bl' | 'br'
    startX: number
    startY: number
    initBox: { x: number; y: number; w: number; h: number }
  } | null>(null)

  const onPointerDown = (e: React.PointerEvent, mode: 'move' | 'tl' | 'tr' | 'bl' | 'br') => {
    e.stopPropagation()
    e.preventDefault()
    try {
      ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
    } catch {}
    setDragState({
      mode,
      startX: e.clientX,
      startY: e.clientY,
      initBox: { ...box },
    })
  }

  const onPointerMove = (e: React.PointerEvent) => {
    if (!dragState || !imgRef.current) return
    const rect = imgRef.current.getBoundingClientRect()
    if (rect.width <= 0 || rect.height <= 0) return

    const dxPercent = ((e.clientX - dragState.startX) / rect.width) * 100
    const dyPercent = ((e.clientY - dragState.startY) / rect.height) * 100
    const init = dragState.initBox

    if (dragState.mode === 'move') {
      const newX = Math.max(0, Math.min(100 - init.w, init.x + dxPercent))
      const newY = Math.max(0, Math.min(100 - init.h, init.y + dyPercent))
      setBox({ ...init, x: newX, y: newY })
    } else if (dragState.mode === 'br') {
      const newW = Math.max(5, Math.min(100 - init.x, init.w + dxPercent))
      const newH = Math.max(5, Math.min(100 - init.y, init.h + dyPercent))
      setBox({ ...init, w: newW, h: newH })
    } else if (dragState.mode === 'tl') {
      const maxDx = init.x + init.w - 5
      const clampedDx = Math.min(dxPercent, maxDx)
      const newW = init.w - clampedDx
      const newX = init.x + clampedDx

      const maxDy = init.y + init.h - 5
      const clampedDy = Math.min(dyPercent, maxDy)
      const newH = init.h - clampedDy
      const newY = init.y + clampedDy
      setBox({ x: Math.max(0, newX), y: Math.max(0, newY), w: newW, h: newH })
    } else if (dragState.mode === 'tr') {
      const newW = Math.max(5, Math.min(100 - init.x, init.w + dxPercent))
      const maxDy = init.y + init.h - 5
      const clampedDy = Math.min(dyPercent, maxDy)
      const newH = init.h - clampedDy
      const newY = init.y + clampedDy
      setBox({ x: init.x, y: Math.max(0, newY), w: newW, h: newH })
    } else if (dragState.mode === 'bl') {
      const maxDx = init.x + init.w - 5
      const clampedDx = Math.min(dxPercent, maxDx)
      const newW = init.w - clampedDx
      const newX = init.x + clampedDx
      const newH = Math.max(5, Math.min(100 - init.y, init.h + dyPercent))
      setBox({ x: Math.max(0, newX), y: init.y, w: newW, h: newH })
    }
  }

  const onPointerUp = (e: React.PointerEvent) => {
    if (dragState) {
      try {
        ;(e.target as HTMLElement).releasePointerCapture(e.pointerId)
      } catch {}
      setDragState(null)
    }
  }

  const handleConfirm = () => {
    if (!naturalSize) {
      onCancel()
      return
    }
    const xmin = Math.round((box.x / 100) * naturalSize.w)
    const ymin = Math.round((box.y / 100) * naturalSize.h)
    const xmax = Math.round(((box.x + box.w) / 100) * naturalSize.w)
    const ymax = Math.round(((box.y + box.h) / 100) * naturalSize.h)
    onConfirm([xmin, ymin, xmax, ymax])
  }

  return (
    <div className="bg-gray-900 text-white rounded-xl p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-xs font-bold text-white">Adjust Face Region</h3>
          <p className="text-[11px] text-gray-400">Drag or resize the box around the face you want to verify.</p>
        </div>
        <button
          onClick={onCancel}
          className="text-gray-400 hover:text-white text-xs px-2.5 py-1 rounded bg-gray-800 hover:bg-gray-700 transition-colors"
        >
          Cancel
        </button>
      </div>

      <div
        className="relative mx-auto overflow-hidden bg-black/60 rounded-lg select-none touch-none flex items-center justify-center p-1 max-h-[360px]"
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
      >
        <div className="relative inline-block max-h-[350px] max-w-full">
          <img
            ref={imgRef}
            src={imageUrl}
            alt="Target face for cropping"
            onLoad={handleImageLoad}
            className="max-h-[350px] w-auto max-w-full object-contain pointer-events-none block rounded"
          />

          <div
            className="absolute border-2 border-emerald-400 bg-emerald-500/20 cursor-move flex items-center justify-center shadow-lg rounded"
            style={{
              left: `${box.x}%`,
              top: `${box.y}%`,
              width: `${box.w}%`,
              height: `${box.h}%`,
            }}
            onPointerDown={(e) => onPointerDown(e, 'move')}
          >
            <span className="text-[10px] font-mono font-bold bg-emerald-600 text-white px-1.5 py-0.5 rounded shadow pointer-events-none">
              Selected Region
            </span>

            <div
              className="absolute -top-1.5 -left-1.5 w-3.5 h-3.5 bg-white border-2 border-emerald-600 rounded-full cursor-nwse-resize hover:scale-125 transition-transform"
              onPointerDown={(e) => onPointerDown(e, 'tl')}
            />
            <div
              className="absolute -top-1.5 -right-1.5 w-3.5 h-3.5 bg-white border-2 border-emerald-600 rounded-full cursor-nesw-resize hover:scale-125 transition-transform"
              onPointerDown={(e) => onPointerDown(e, 'tr')}
            />
            <div
              className="absolute -bottom-1.5 -left-1.5 w-3.5 h-3.5 bg-white border-2 border-emerald-600 rounded-full cursor-nesw-resize hover:scale-125 transition-transform"
              onPointerDown={(e) => onPointerDown(e, 'bl')}
            />
            <div
              className="absolute -bottom-1.5 -right-1.5 w-3.5 h-3.5 bg-white border-2 border-emerald-600 rounded-full cursor-nwse-resize hover:scale-125 transition-transform"
              onPointerDown={(e) => onPointerDown(e, 'br')}
            />
          </div>
        </div>
      </div>

      <div className="flex items-center justify-between pt-1">
        <span className="text-[11px] text-gray-400 font-mono">
          Box: ({box.x.toFixed(0)}%, {box.y.toFixed(0)}%) - {box.w.toFixed(0)}x{box.h.toFixed(0)}%
        </span>
        <div className="flex items-center gap-2">
          <Button variant="outline" onClick={onCancel} className="text-white border-gray-700 hover:bg-gray-800">
            Cancel
          </Button>
          <Button onClick={handleConfirm} className="bg-emerald-600 hover:bg-emerald-500 text-white">
            Use Selected Region
          </Button>
        </div>
      </div>
    </div>
  )
}

// ─── 1. Face Input Component ──────────────────────────────────────────────────

function FaceInputCard({
  imageUrl,
  imageFilename,
  cropBox,
  onCropBoxChange,
  onImageSelect,
  onClear,
  onRun,
  pipelineState,
  apiResult
}: {
  imageUrl: string | null
  imageFilename: string
  cropBox: number[] | null
  onCropBoxChange: (box: number[] | null) => void
  onImageSelect: (url: string, file?: File) => void
  onClear: () => void
  onRun: () => void
  pipelineState: PipelineState
  apiResult: ApiResultData | null
}) {
  const [dragging, setDragging] = useState(false)
  const [isAdjusting, setIsAdjusting] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const thumbImgRef = useRef<HTMLImageElement>(null)
  const [naturalSize, setNaturalSize] = useState<{ w: number; h: number } | null>(null)

  const handleThumbImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const nw = e.currentTarget.naturalWidth
    const nh = e.currentTarget.naturalHeight
    if (nw > 0 && nh > 0) {
      setNaturalSize({ w: nw, h: nh })
    }
  }

  useEffect(() => {
    if (thumbImgRef.current && thumbImgRef.current.complete) {
      const nw = thumbImgRef.current.naturalWidth
      const nh = thumbImgRef.current.naturalHeight
      if (nw > 0 && nh > 0) {
        setNaturalSize({ w: nw, h: nh })
      }
    }
  }, [imageUrl])

  const handleFile = (file: File) => {
    if (!file.type.match(/image\/(jpeg|jpg|png|webp)/)) return
    const url = URL.createObjectURL(file)
    onImageSelect(url, file)
  }

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }, [])

  const onDragOver = (e: React.DragEvent) => { e.preventDefault(); setDragging(true) }
  const onDragLeave = () => setDragging(false)

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
  }

  const isRunning = pipelineState === 'loading'
  const faceDetected = apiResult?.face_detected === true
  const detConfidence = apiResult?.detection_score ? apiResult.detection_score.toFixed(4) : '--'
  const embeddingGenerated = apiResult?.embedding_generated === true

  // Bounding box to display on thumbnail: custom region if set, otherwise automatic InsightFace box
  const activeBbox = cropBox || (faceDetected && apiResult?.bbox ? apiResult.bbox : null)

  return (
    <Card className="p-4">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h2 className="text-sm font-bold text-[#111827]">Face Input</h2>
          <p className="text-xs text-[#667085]">Upload an image containing a detectable face.</p>
        </div>
        {imageFilename && (
          <span
            className="text-xs font-mono text-[#667085] bg-gray-100 px-2.5 py-1 rounded-md border border-[#E5E7EB] max-w-[220px] truncate"
            title={imageFilename}
          >
            {truncateFilename(imageFilename, 26)}
          </span>
        )}
      </div>

      {!imageUrl ? (
        <div
          className={cn(
            'border-2 border-dashed rounded-xl flex flex-col items-center justify-center gap-2 py-8 cursor-pointer transition-all duration-200',
            dragging ? 'border-[#111827] bg-gray-50' : 'border-[#E5E7EB] hover:border-gray-300 hover:bg-gray-50/50'
          )}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onClick={() => inputRef.current?.click()}
        >
          <IconUpload />
          <div className="text-center">
            <p className="text-sm font-semibold text-[#111827]">Drop image here or click to browse</p>
            <p className="text-xs text-[#667085]">JPG, JPEG, PNG or WEBP (Max 10 MB)</p>
          </div>
          <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png,.webp" className="hidden" onChange={onFileChange} />
        </div>
      ) : isAdjusting ? (
        <RegionAdjuster
          imageUrl={imageUrl}
          initialBbox={cropBox || apiResult?.bbox || null}
          onConfirm={(box) => {
            onCropBoxChange(box)
            setIsAdjusting(false)
          }}
          onCancel={() => setIsAdjusting(false)}
        />
      ) : (
        <div className="flex flex-col sm:flex-row gap-4 items-center">
          <div className="relative w-32 h-32 rounded-xl overflow-hidden bg-gray-900 flex-shrink-0 border border-[#E5E7EB] flex items-center justify-center p-0.5">
            <div className="relative max-w-full max-h-full flex items-center justify-center">
              <img
                ref={thumbImgRef}
                src={imageUrl}
                alt="Uploaded face"
                onLoad={handleThumbImageLoad}
                className="max-w-32 max-h-32 w-auto h-auto object-contain rounded-lg block"
              />
              {activeBbox && activeBbox.length === 4 && naturalSize && naturalSize.w > 0 && naturalSize.h > 0 && (
                <div
                  className="absolute border-2 border-[#059669] bg-[#059669]/20 rounded pointer-events-none transition-all duration-150 shadow-sm"
                  style={{
                    left: `${(activeBbox[0] / naturalSize.w) * 100}%`,
                    top: `${(activeBbox[1] / naturalSize.h) * 100}%`,
                    width: `${((activeBbox[2] - activeBbox[0]) / naturalSize.w) * 100}%`,
                    height: `${((activeBbox[3] - activeBbox[1]) / naturalSize.h) * 100}%`,
                  }}
                />
              )}
            </div>
          </div>

          <div className="flex-1 w-full space-y-3">
            <div className="grid grid-cols-2 gap-2">
              <div className="bg-[#F7F8FA] rounded-lg p-2.5 border border-[#E5E7EB]">
                <p className="text-[10px] font-semibold text-[#98A2B3] uppercase tracking-wider mb-0.5">Detection Score</p>
                <p className="font-mono text-xs font-bold text-[#111827]">{detConfidence}</p>
              </div>
              <div className="bg-[#F7F8FA] rounded-lg p-2.5 border border-[#E5E7EB]">
                <p className="text-[10px] font-semibold text-[#98A2B3] uppercase tracking-wider mb-0.5">512D Embedding</p>
                <p className={cn('font-mono text-xs font-bold', embeddingGenerated ? 'text-[#059669]' : 'text-[#98A2B3]')}>
                  {embeddingGenerated ? 'Generated' : '--'}
                </p>
              </div>
            </div>

            {cropBox && (
              <div className="flex items-center justify-between bg-emerald-50 border border-emerald-200 text-emerald-800 px-2.5 py-1.5 rounded-lg text-xs">
                <span className="font-medium flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                  Custom Face Region Selected ({cropBox.join(', ')})
                </span>
                <button
                  onClick={() => onCropBoxChange(null)}
                  className="text-emerald-700 hover:text-emerald-900 font-semibold text-[11px] underline"
                >
                  Reset Region
                </button>
              </div>
            )}

            <div className="flex flex-wrap gap-2">
              <Button onClick={onRun} disabled={!imageUrl || isRunning} className="flex-1 min-w-[120px]">
                {isRunning ? (
                  <>
                    <IconSpinner />
                    Verifying...
                  </>
                ) : (
                  'Run Verification'
                )}
              </Button>
              <Button variant="outline" onClick={() => setIsAdjusting(true)} disabled={isRunning}>
                Adjust Face Region
              </Button>
              <Button variant="outline" onClick={onClear} disabled={isRunning}>
                Clear
              </Button>
            </div>
          </div>
        </div>
      )}
    </Card>
  )
}

// ─── 2. Pipeline Stepper Component (With Filled Green Checkmark Circles) ─────

function PipelineStepper({ steps }: { steps: PipelineStep[] }) {
  return (
    <Card className="p-3.5">
      <div className="flex items-center justify-between relative px-4">
        {/* Render segmented connecting lines */}
        <div className="absolute left-8 right-8 top-[16px] flex items-center justify-between z-0 pointer-events-none">
          {steps.slice(0, steps.length - 1).map((step, i) => {
            const nextStep = steps[i + 1]
            const isCompletedSegment = step.state === 'done' && (nextStep.state === 'done' || nextStep.state === 'active')
            const isFailedSegment = step.state === 'done' && nextStep.state === 'failed'
            const isActiveSegment = step.state === 'active'

            return (
              <div
                key={i}
                className={cn(
                  'h-0.5 flex-1 transition-all duration-300 mx-1',
                  isCompletedSegment && 'bg-[#059669]',
                  isFailedSegment && 'bg-red-500',
                  isActiveSegment && 'bg-[#2563EB]',
                  !isCompletedSegment && !isFailedSegment && !isActiveSegment && 'bg-[#E5E7EB]'
                )}
              />
            )
          })}
        </div>

        {/* Step Nodes */}
        {steps.map((step, i) => {
          const isDone = step.state === 'done'
          const isActive = step.state === 'active'
          const isFailed = step.state === 'failed'
          const isSkipped = step.state === 'skipped'

          return (
            <div key={step.id} className="flex flex-col items-center gap-1.5 relative z-10 flex-1">
              <div
                className={cn(
                  'w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold transition-all duration-300 border-2 shadow-sm',
                  isDone && 'bg-[#059669] border-[#059669] text-white',
                  isActive && 'bg-white border-[#2563EB] text-[#2563EB] ring-4 ring-blue-50',
                  isFailed && 'bg-red-600 border-red-600 text-white',
                  isSkipped && 'bg-gray-100 border-[#E5E7EB] text-[#98A2B3]',
                  !isDone && !isActive && !isFailed && !isSkipped && 'bg-white border-[#E5E7EB] text-[#98A2B3]'
                )}
              >
                {isDone ? (
                  <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                    <path d="M3 8.5l3.5 3.5 6.5-7" stroke="white" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                ) : isFailed ? (
                  <svg width="12" height="12" viewBox="0 0 14 14" fill="none">
                    <path d="M3 3l8 8M11 3l-8 8" stroke="white" strokeWidth="2.5" strokeLinecap="round" />
                  </svg>
                ) : isSkipped ? (
                  <span className="font-mono text-xs font-semibold text-gray-400">-</span>
                ) : isActive ? (
                  <span className="w-2.5 h-2.5 rounded-full bg-[#2563EB] animate-pulse" />
                ) : (
                  <span className="font-mono text-xs font-semibold">{i + 1}</span>
                )}
              </div>

              <div className="text-center">
                <p
                  className={cn(
                    'text-[11px] font-bold leading-tight',
                    isDone && 'text-[#059669]',
                    isActive && 'text-[#2563EB]',
                    isFailed && 'text-red-600',
                    (isSkipped || (!isDone && !isActive && !isFailed)) && 'text-[#98A2B3]'
                  )}
                >
                  {step.label}
                </p>
                <p className="text-[9px] text-[#98A2B3] leading-none mt-0.5">
                  {isFailed ? 'Failed' : isSkipped ? 'Not reached' : step.sublabel}
                </p>
              </div>
            </div>
          )
        })}
      </div>
    </Card>
  )
}

// ─── Calm Dynamic Loading Progress Component ──────────────────────────────────

function DynamicLoadingState({
  currentStep,
  elapsedSeconds,
  candidateCount
}: {
  currentStep: number
  elapsedSeconds: number
  candidateCount?: number
}) {
  let title = 'Processing pipeline...'
  let description = 'Evaluating image and searching visual databases.'
  let helper = 'Initial processing usually takes a few seconds.'
  let secondary = 'Please keep this window open while verification completes.'

  if (currentStep === 0) {
    title = 'Detecting face...'
    description = 'Analyzing the uploaded image.'
    helper = 'Initial processing usually takes a few seconds.'
    secondary = 'Detecting face landmarks and orientation.'
  } else if (currentStep === 1) {
    title = 'Generating face embedding...'
    description = 'Creating a 512-dimensional face representation.'
    helper = 'Preparing the face for visual verification.'
    secondary = 'Normalizing vector space embeddings.'
  } else if (currentStep === 2) {
    title = 'Searching the open web...'
    description = 'Finding visual matches with Google Lens.'
    helper = 'Typical verification time: 2–3 minutes.'
    if (elapsedSeconds > 45) {
      secondary = 'Search is taking a little longer than usual. Open-web search can take a few minutes depending on the number of results.'
    } else {
      secondary = 'Open-web search and candidate verification may take a few minutes.'
    }
  } else if (currentStep === 3) {
    title = 'Verifying candidates...'
    description = 'Comparing the detected face against discovered web candidates.'
    helper = candidateCount && candidateCount > 0
      ? `${candidateCount} candidates discovered`
      : 'Evaluating visual similarity against cutoff threshold.'
    secondary = 'Running L2-normalized cosine similarity matching.'
  } else if (currentStep >= 4) {
    title = 'Registering blockchain proof...'
    description = 'Creating the SHA-256 fingerprint and recording the verified result.'
    helper = 'Finalizing verification proof.'
    secondary = 'Performing final on-chain integrity check.'
  }

  return (
    <Card className="p-4 border-blue-200 bg-blue-50/40">
      <div className="flex items-start gap-3.5">
        <div className="w-9 h-9 rounded-xl bg-blue-100 flex items-center justify-center flex-shrink-0 text-blue-600 mt-0.5">
          <IconSpinner />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-3 mb-1">
            <span className="text-[11px] font-bold text-blue-900 tracking-wide uppercase">
              VERIFICATION IN PROGRESS
            </span>
            <span className="text-[11px] font-semibold text-blue-800 bg-blue-100/90 px-2.5 py-0.5 rounded-full">
              Typical time: 2–3 minutes
            </span>
          </div>

          <h3 className="text-sm font-bold text-[#111827] mb-0.5">{title}</h3>
          <p className="text-xs text-[#667085] mb-2">{description}</p>

          <div className="pt-2 border-t border-blue-100/80 space-y-1 text-[11px] text-blue-900">
            <div className="flex items-center gap-1.5 font-semibold">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse" />
              <span>{helper}</span>
            </div>
            {secondary && <p className="text-[10px] text-blue-700/90 pl-3 leading-relaxed">{secondary}</p>}
          </div>
        </div>
      </div>
    </Card>
  )
}

// ─── Search Failure / Timeout Card ─────────────────────────────────────────────

function SearchErrorCard({ result, onRetry }: { result: ApiResultData; onRetry: () => void }) {
  const threshold = result.threshold ? result.threshold.toFixed(4) : '0.5000'
  const errMessage = result.error?.replace(/^Search failed:\s*/i, '') || 'Google Lens search timed out before candidates could be retrieved.'

  return (
    <Card className="p-5 border-amber-300 bg-amber-50/50">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded-md bg-amber-500 flex items-center justify-center text-white font-bold text-xs">
            ⚠
          </div>
          <span className="text-xs font-bold text-amber-900 tracking-wide uppercase">
            VERIFICATION COULD NOT COMPLETE
          </span>
        </div>
        <StatusPill status="warning" label="SEARCH TIMEOUT" />
      </div>

      <h3 className="text-sm font-bold text-amber-950 mb-1">
        Open-web search could not be completed.
      </h3>
      <p className="text-xs text-amber-900 mb-3 leading-relaxed">
        {errMessage} The query face was detected successfully, but visual candidate discovery on Google Lens timed out before results could be evaluated.
      </p>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 bg-white p-3 rounded-lg border border-amber-200 mb-3">
        <div>
          <p className="text-[10px] font-semibold text-amber-700 uppercase tracking-wider mb-0.5">Candidates</p>
          <p className="font-mono text-sm font-bold text-amber-950">—</p>
        </div>
        <div>
          <p className="text-[10px] font-semibold text-amber-700 uppercase tracking-wider mb-0.5">Best Similarity</p>
          <p className="font-mono text-sm font-bold text-amber-950">—</p>
        </div>
        <div>
          <p className="text-[10px] font-semibold text-amber-700 uppercase tracking-wider mb-0.5">Required Cutoff</p>
          <p className="font-mono text-sm font-bold text-amber-950">{threshold}</p>
        </div>
        <div>
          <p className="text-[10px] font-semibold text-amber-700 uppercase tracking-wider mb-0.5">Blockchain</p>
          <p className="font-mono text-sm font-bold text-amber-900">Not registered</p>
        </div>
      </div>

      <div className="flex items-center justify-between pt-2 border-t border-amber-200/80 text-xs">
        <p className="text-amber-800 text-[11px]">
          Please check your network connection and click below to try verification again.
        </p>
        <Button onClick={onRetry} variant="primary" className="py-1.5 px-3">
          Try Verification Again
        </Button>
      </div>
    </Card>
  )
}

// ─── 3. Primary Candidate Result Components ───────────────────────────────────

function VerifiedCandidateCard({ result }: { result: ApiResultData }) {
  const cand = result.candidate
  if (!cand) return null

  const similarity = cand.similarity ? cand.similarity.toFixed(4) : '--'
  const threshold = result.threshold ? result.threshold.toFixed(4) : '0.5000'
  const domain = cand.domain || 'web'
  const title = cand.title || 'Verified Web Candidate'
  const sourceUrl = cand.source_url || '#'
  const candidateImg = cand.candidate_image_url

  return (
    <Card className="p-5 border-2 border-emerald-500/30 bg-gradient-to-br from-white via-white to-emerald-50/20 shadow-sm">
      <div className="flex items-center justify-between gap-4 mb-3">
        <div className="flex items-center gap-2 flex-wrap">
          <div className="w-5 h-5 rounded bg-emerald-600 flex items-center justify-center text-white font-bold text-xs">
            ✓
          </div>
          <span className="text-xs font-bold text-emerald-800 tracking-wide uppercase">VERIFICATION COMPLETE</span>
        </div>
        <StatusPill status="success" label="VERIFIED MATCH" />
      </div>

      <div className="flex flex-col sm:flex-row gap-4">
        {candidateImg && (
          <div className="w-28 h-32 rounded-xl bg-gray-100 flex-shrink-0 overflow-hidden border border-[#E5E7EB]">
            <img
              src={candidateImg}
              alt="Matched candidate"
              className="w-full h-full object-cover"
              onError={(e) => {
                (e.target as HTMLImageElement).style.display = 'none'
              }}
            />
          </div>
        )}
        <div className="flex-1 min-w-0 flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-bold text-[#111827] bg-gray-100 px-2 py-0.5 rounded border border-[#E5E7EB]">{domain}</span>
              {sourceUrl !== '#' && (
                <a
                  href={sourceUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="text-xs text-[#2563EB] hover:underline inline-flex items-center gap-1 font-semibold"
                >
                  Open Result Source
                  <svg width="10" height="10" viewBox="0 0 10 10" fill="none"><path d="M2 8L8 2M8 2H5M8 2v3" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" /></svg>
                </a>
              )}
            </div>
            <h3 className="text-sm font-bold text-[#111827] mb-1 leading-snug">
              {title}
            </h3>
          </div>

          <div className="pt-3 border-t border-[#E5E7EB] mt-2">
            <div className="flex items-center gap-6">
              <div>
                <p className="text-[10px] font-semibold text-[#98A2B3] uppercase tracking-wider mb-0.5">Face Similarity</p>
                <p className="font-mono text-xl font-extrabold text-[#059669]">{similarity}</p>
              </div>
              <div>
                <p className="text-[10px] font-semibold text-[#98A2B3] uppercase tracking-wider mb-0.5">Threshold</p>
                <p className="font-mono text-xl font-bold text-[#111827]">{threshold}</p>
              </div>
              <div className="flex-1 hidden sm:block">
                <div className="h-2 bg-[#E5E7EB] rounded-full overflow-hidden">
                  <div
                    className="h-2 bg-[#059669] rounded-full transition-all duration-500"
                    style={{ width: `${Math.min(100, (parseFloat(similarity) || 0) * 100)}%` }}
                  />
                </div>
                <p className="text-[10px] text-[#059669] font-semibold mt-1">✓ Candidate exceeds verification threshold</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Card>
  )
}

function UnverifiedCandidateCard({ result }: { result: ApiResultData }) {
  const evaluated = result.candidates_evaluated ?? 0
  const highestSim = result.highest_similarity ? result.highest_similarity.toFixed(4) : '--'
  const threshold = result.threshold ? result.threshold.toFixed(4) : '0.5000'
  const faceDetected = result.face_detected === true

  return (
    <Card className="p-5 border-amber-200 bg-amber-50/40">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2 flex-wrap">
          <div className="w-5 h-5 rounded bg-amber-500 flex items-center justify-center text-white font-bold text-xs">
            !
          </div>
          <span className="text-xs font-bold text-amber-900 tracking-wide uppercase">NO VERIFIED MATCH</span>
        </div>
        <StatusPill status="warning" label="NO MATCH" />
      </div>

      <p className="text-xs font-semibold text-amber-950 mb-3">
        {evaluated === 0
          ? 'No matching web candidates were found.'
          : 'Candidates were found, but no candidate exceeded the verification threshold.'}
      </p>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 bg-white p-3 rounded-lg border border-amber-200/80 mb-2">
        <div>
          <p className="text-[10px] font-semibold text-amber-700 uppercase tracking-wider mb-0.5">Evaluated</p>
          <p className="font-mono text-sm font-bold text-amber-950">{evaluated} candidates</p>
        </div>
        <div>
          <p className="text-[10px] font-semibold text-amber-700 uppercase tracking-wider mb-0.5">Best Similarity</p>
          <p className="font-mono text-sm font-bold text-amber-950">{highestSim}</p>
        </div>
        <div>
          <p className="text-[10px] font-semibold text-amber-700 uppercase tracking-wider mb-0.5">Required Cutoff</p>
          <p className="font-mono text-sm font-bold text-amber-950">{threshold}</p>
        </div>
        <div>
          <p className="text-[10px] font-semibold text-amber-700 uppercase tracking-wider mb-0.5">Blockchain Record</p>
          <p className="font-mono text-sm font-bold text-amber-950">None</p>
        </div>
      </div>

      <p className="text-[11px] text-amber-800">
        {!faceDetected
          ? 'No face was detected in the input image.'
          : 'No candidate image surpassed the threshold. No blockchain record was registered.'}
      </p>
    </Card>
  )
}

// ─── 4. Metrics Row Component ─────────────────────────────────────────────────

function MetricsRow({ result }: { result: ApiResultData }) {
  const isSearchFailed = result.search_status === 'failed' || result.error_stage === 'web_search'
  const evaluated = isSearchFailed ? '—' : (result.candidates_evaluated ?? 0)
  const faceDetected = result.face_detected ? '1' : '0'
  const similarity = isSearchFailed
    ? '—'
    : (result.candidate?.similarity
        ? result.candidate.similarity.toFixed(4)
        : (result.highest_similarity ? result.highest_similarity.toFixed(4) : '--'))
  const threshold = result.threshold ? result.threshold.toFixed(4) : '0.5000'

  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
      <MetricCard label="Candidates" value={evaluated} />
      <MetricCard label="Faces Detected" value={faceDetected} />
      <MetricCard label="Best Similarity" value={similarity} sub="Score" />
      <MetricCard label="Threshold" value={threshold} sub="Required" />
    </div>
  )
}

// ─── 6. Blockchain Proof Component ────────────────────────────────────────────

function BlockchainProofCard({ blockchain }: { blockchain: ApiResultData['blockchain'] }) {
  if (!blockchain) return null

  const fingerprint = blockchain.fingerprint || '--'
  const recordId = blockchain.record_id || blockchain.block_hash || '--'
  const previousHash = blockchain.previous_hash || '--'
  const integrity = blockchain.chain_integrity || 'PASSED'
  const reverification = blockchain.reverification || 'VERIFIED'

  return (
    <Card className="p-4 border border-emerald-100 bg-gradient-to-r from-white to-emerald-50/20">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded-md bg-emerald-100 flex items-center justify-center">
            <IconShield />
          </div>
          <div>
            <h3 className="text-xs font-bold text-[#111827]">Blockchain Proof</h3>
            <p className="text-[10px] text-[#667085]">SHA-256 tamper-evident ledger registration</p>
          </div>
        </div>
        <StatusPill status="success" label="REGISTERED" />
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 mb-3">
        <div className="bg-white rounded-lg p-2.5 border border-[#E5E7EB] flex items-center justify-between">
          <div className="min-w-0 pr-2">
            <p className="text-[10px] font-semibold text-[#98A2B3] uppercase tracking-wider mb-0.5">SHA-256 Fingerprint</p>
            <p className="font-mono text-xs font-bold text-[#111827] truncate" title={fingerprint}>
              {truncateHash(fingerprint, 10, 8)}
            </p>
          </div>
          <IconCopy textToCopy={fingerprint} label="Fingerprint" />
        </div>

        <div className="bg-white rounded-lg p-2.5 border border-[#E5E7EB] flex items-center justify-between">
          <div className="min-w-0 pr-2">
            <p className="text-[10px] font-semibold text-[#98A2B3] uppercase tracking-wider mb-0.5">Block Hash</p>
            <p className="font-mono text-xs font-bold text-[#111827] truncate" title={recordId}>
              {truncateHash(recordId, 10, 8)}
            </p>
          </div>
          <IconCopy textToCopy={recordId} label="Block Hash" />
        </div>

        <div className="bg-white rounded-lg p-2.5 border border-[#E5E7EB] flex items-center justify-between">
          <div className="min-w-0 pr-2">
            <p className="text-[10px] font-semibold text-[#98A2B3] uppercase tracking-wider mb-0.5">Previous Hash</p>
            <p className="font-mono text-xs font-bold text-[#111827] truncate" title={previousHash}>
              {truncateHash(previousHash, 10, 8)}
            </p>
          </div>
          <IconCopy textToCopy={previousHash} label="Previous Hash" />
        </div>
      </div>

      <div className="flex items-center justify-between pt-2.5 border-t border-[#E5E7EB]/80 text-xs">
        <div className="flex items-center gap-5">
          <div className="flex items-center gap-1.5">
            <span className="text-[11px] text-[#667085]">Chain Integrity:</span>
            <span className="font-mono font-bold text-[11px] text-[#059669]">{integrity}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-[11px] text-[#667085]">On-Chain Re-verification:</span>
            <span className="font-mono font-bold text-[11px] text-[#059669]">{reverification}</span>
          </div>
        </div>

        <div className="hidden sm:flex items-center gap-1 text-[10px] text-[#98A2B3] font-mono">
          <span>Genesis</span>
          <span>→</span>
          <span>Block</span>
          <span>→</span>
          <span className="font-bold text-[#059669]">Record</span>
        </div>
      </div>
    </Card>
  )
}

// ─── 8. Pipeline Activity Log Component ───────────────────────────────────────

function PipelineActivityLog({ entries }: { entries: LogEntry[] }) {
  const [isExpanded, setIsExpanded] = useState(false)

  if (!entries || entries.length === 0) return null

  const dotColors = {
    info: 'bg-blue-400',
    success: 'bg-emerald-500',
    warning: 'bg-amber-500',
    error: 'bg-red-500',
  }
  const textColors = {
    info: 'text-[#667085]',
    success: 'text-[#059669]',
    warning: 'text-amber-600',
    error: 'text-red-600',
  }

  return (
    <Card className="p-3.5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h3 className="text-xs font-bold text-[#111827]">Pipeline Activity</h3>
          <span className="text-[10px] font-medium text-[#667085] bg-gray-100 px-2 py-0.5 rounded-full font-mono">
            {entries.length} events
          </span>
        </div>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="text-xs font-semibold text-[#2563EB] hover:text-blue-800 transition-colors flex items-center gap-1"
        >
          {isExpanded ? 'Collapse Logs' : 'Expand Logs'}
          <svg
            width="12"
            height="12"
            viewBox="0 0 12 12"
            fill="none"
            className={cn('transition-transform duration-200', isExpanded && 'rotate-180')}
          >
            <path d="M2.5 4.5l3.5 3.5 3.5-3.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
      </div>

      {isExpanded && (
        <div className="mt-3 bg-[#F7F8FA] rounded-lg border border-[#E5E7EB] p-3 max-h-56 overflow-y-auto space-y-1.5">
          {entries.map((entry, i) => (
            <div key={i} className="flex items-start gap-2.5">
              <span className={cn('w-1.5 h-1.5 rounded-full mt-[5px] flex-shrink-0', dotColors[entry.status || 'info'])} />
              <span className="font-mono text-[10px] text-[#98A2B3] flex-shrink-0">[{entry.time || formatNow()}]</span>
              <span className={cn('font-mono text-[11px] leading-snug', textColors[entry.status || 'info'])}>{entry.message}</span>
            </div>
          ))}
        </div>
      )}
    </Card>
  )
}

// ─── Error Card ───────────────────────────────────────────────────────────────

function ErrorCard({ message }: { message: string }) {
  return (
    <Card className="p-4 border-red-200 bg-red-50">
      <div className="flex items-center gap-3 text-red-800">
        <svg width="18" height="18" viewBox="0 0 20 20" fill="none">
          <circle cx="10" cy="10" r="9" stroke="#DC2626" strokeWidth="1.5" />
          <path d="M10 6v5M10 14v.5" stroke="#DC2626" strokeWidth="1.5" strokeLinecap="round" />
        </svg>
        <div>
          <p className="text-xs font-bold">Verification Request Failed</p>
          <p className="text-xs text-red-700 mt-0.5">{message}</p>
        </div>
      </div>
    </Card>
  )
}

// ─── Verification Page Component ──────────────────────────────────────────────

function VerificationPage({ onRecordSuccess }: { onRecordSuccess: (record: HistoryRecord) => void }) {
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [imageFilename, setImageFilename] = useState('')
  const [confirmedCropBox, setConfirmedCropBox] = useState<number[] | null>(null)
  const [pipelineState, setPipelineState] = useState<PipelineState>('idle')
  const [steps, setSteps] = useState<PipelineStep[]>(PIPELINE_STEPS_INITIAL)
  const [logEntries, setLogEntries] = useState<LogEntry[]>([])
  const [loadingStep, setLoadingStep] = useState(0)
  const [apiResult, setApiResult] = useState<ApiResultData | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  // Internal duration tracking (not exposed as a stopwatch in primary UI)
  const [elapsedSeconds, setElapsedSeconds] = useState(0)
  const timerRef = useRef<number | null>(null)
  const startTimeRef = useRef<number>(0)

  const startTimer = () => {
    setElapsedSeconds(0)
    startTimeRef.current = Date.now()
    if (timerRef.current) clearInterval(timerRef.current)
    timerRef.current = window.setInterval(() => {
      const elapsed = Math.floor((Date.now() - startTimeRef.current) / 1000)
      setElapsedSeconds(elapsed)
    }, 1000)
  }

  const stopTimer = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
  }

  const resetTimer = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
    setElapsedSeconds(0)
  }

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [])

  const resetPipeline = () => {
    setSteps(PIPELINE_STEPS_INITIAL)
    setLogEntries([])
    setLoadingStep(0)
    setApiResult(null)
    setErrorMessage(null)
  }

  const handleImageSelect = (url: string, file?: File) => {
    setImageUrl(url)
    setSelectedFile(file || null)
    setConfirmedCropBox(null)
    if (file) {
      setImageFilename(file.name)
    } else {
      setImageFilename('uploaded_image.jpg')
    }
    resetPipeline()
    resetTimer()
    setPipelineState('idle')
  }

  const handleClear = () => {
    resetTimer()
    setImageUrl(null)
    setSelectedFile(null)
    setImageFilename('')
    setConfirmedCropBox(null)
    setApiResult(null)
    setErrorMessage(null)
    setLogEntries([])
    setSteps(PIPELINE_STEPS_INITIAL)
    setPipelineState('idle')
  }

  const handleRun = async () => {
    if (!imageUrl) return

    resetPipeline()
    setPipelineState('loading')
    setErrorMessage(null)
    startTimer()

    const formData = new FormData()
    if (selectedFile) {
      formData.append('file', selectedFile)
    } else {
      formData.append('image_name', imageFilename || 'uploaded_image.jpg')
    }
    formData.append('provider', 'web')
    formData.append('threshold', '0.5')
    if (confirmedCropBox && confirmedCropBox.length === 4) {
      formData.append('crop_box', confirmedCropBox.join(','))
    }

    try {
      setLoadingStep(0)
      setSteps(prev => prev.map((s, i) => i === 0 ? { ...s, state: 'active' } : s))

      const res = await fetch('/api/verify', {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        let errDetail = `Server returned HTTP ${res.status}`
        try {
          const errData = await res.json()
          if (errData.detail) errDetail = errData.detail
        } catch {}
        throw new Error(errDetail)
      }

      const data: ApiResultData = await res.json()
      setApiResult(data)

      const logs = data.logs || []
      const isSearchFailed = data.search_status === 'failed' || data.error_stage === 'web_search' || (!data.success && data.error?.toLowerCase().includes('search failed'))
      const hasMatch = data.match_found && data.success

      if (logs.length > 0) {
        const stepSequence = [0, 1, 2, 2, 3, 4]
        logs.forEach((entry, i) => {
          setTimeout(() => {
            const stepIdx = stepSequence[Math.min(i, stepSequence.length - 1)]
            setLoadingStep(stepIdx)
            setLogEntries(prev => [...prev, entry])
            setSteps(prev => prev.map((s, si) => {
              if (si < stepIdx) return { ...s, state: 'done' }
              if (si === stepIdx) return { ...s, state: 'active' }
              return { ...s, state: 'pending' }
            }))
          }, i * 180)
        })

        setTimeout(() => {
          stopTimer()

          if (isSearchFailed) {
            setSteps([
              { id: 1, label: 'Face Detection', sublabel: 'InsightFace', state: 'done' },
              { id: 2, label: 'Search', sublabel: 'Google Lens', state: 'failed' },
              { id: 3, label: 'Verify', sublabel: 'Cosine Match', state: 'skipped' },
              { id: 4, label: 'Blockchain', sublabel: 'SHA-256 Ledger', state: 'skipped' },
              { id: 5, label: 'Re-Verify', sublabel: 'On-Chain Audit', state: 'skipped' },
            ])
            setPipelineState('search_error')
          } else if (hasMatch) {
            setSteps([
              { id: 1, label: 'Face Detection', sublabel: 'InsightFace', state: 'done' },
              { id: 2, label: 'Search', sublabel: 'Google Lens', state: 'done' },
              { id: 3, label: 'Verify', sublabel: 'Cosine Match', state: 'done' },
              { id: 4, label: 'Blockchain', sublabel: 'SHA-256 Ledger', state: 'done' },
              { id: 5, label: 'Re-Verify', sublabel: 'On-Chain Audit', state: 'done' },
            ])
            setPipelineState('success')

            // Record session history
            onRecordSuccess({
              id: `rec-${Date.now()}`,
              date: new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }),
              input: imageFilename || 'uploaded_image.jpg',
              candidates: data.candidates_evaluated,
              similarity: data.candidate?.similarity ? data.candidate.similarity.toFixed(4) : (data.highest_similarity?.toFixed(4) || '--'),
              source: data.candidate?.domain || 'web',
              blockchain: 'Verified',
              status: 'success'
            })
          } else {
            // Search succeeded, but no match surpassed threshold
            setSteps([
              { id: 1, label: 'Face Detection', sublabel: 'InsightFace', state: 'done' },
              { id: 2, label: 'Search', sublabel: 'Google Lens', state: 'done' },
              { id: 3, label: 'Verify', sublabel: 'Cosine Match', state: 'done' },
              { id: 4, label: 'Blockchain', sublabel: 'SHA-256 Ledger', state: 'skipped' },
              { id: 5, label: 'Re-Verify', sublabel: 'On-Chain Audit', state: 'skipped' },
            ])
            setPipelineState('failure')

            onRecordSuccess({
              id: `rec-${Date.now()}`,
              date: new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }),
              input: imageFilename || 'uploaded_image.jpg',
              candidates: data.candidates_evaluated,
              similarity: data.highest_similarity ? data.highest_similarity.toFixed(4) : '--',
              source: 'web',
              blockchain: 'None',
              status: 'failure'
            })
          }
        }, logs.length * 180 + 250)
      } else {
        stopTimer()
        if (isSearchFailed) {
          setSteps([
            { id: 1, label: 'Face Detection', sublabel: 'InsightFace', state: 'done' },
            { id: 2, label: 'Search', sublabel: 'Google Lens', state: 'failed' },
            { id: 3, label: 'Verify', sublabel: 'Cosine Match', state: 'skipped' },
            { id: 4, label: 'Blockchain', sublabel: 'SHA-256 Ledger', state: 'skipped' },
            { id: 5, label: 'Re-Verify', sublabel: 'On-Chain Audit', state: 'skipped' },
          ])
          setPipelineState('search_error')
        } else if (hasMatch) {
          setSteps([
            { id: 1, label: 'Face Detection', sublabel: 'InsightFace', state: 'done' },
            { id: 2, label: 'Search', sublabel: 'Google Lens', state: 'done' },
            { id: 3, label: 'Verify', sublabel: 'Cosine Match', state: 'done' },
            { id: 4, label: 'Blockchain', sublabel: 'SHA-256 Ledger', state: 'done' },
            { id: 5, label: 'Re-Verify', sublabel: 'On-Chain Audit', state: 'done' },
          ])
          setPipelineState('success')
        } else {
          setSteps([
            { id: 1, label: 'Face Detection', sublabel: 'InsightFace', state: 'done' },
            { id: 2, label: 'Search', sublabel: 'Google Lens', state: 'done' },
            { id: 3, label: 'Verify', sublabel: 'Cosine Match', state: 'done' },
            { id: 4, label: 'Blockchain', sublabel: 'SHA-256 Ledger', state: 'skipped' },
            { id: 5, label: 'Re-Verify', sublabel: 'On-Chain Audit', state: 'skipped' },
          ])
          setPipelineState('failure')
        }
      }
    } catch (err: any) {
      console.error('Verification error:', err)
      stopTimer()
      setErrorMessage(err.message || 'Failed to connect to verification backend.')
      setPipelineState('error')
      setSteps(prev => prev.map(s => s.state === 'active' ? { ...s, state: 'failed' } : s))
    }
  }

  return (
    <div className="space-y-4 max-w-3xl mx-auto">
      {/* 1. Face Input Card */}
      <FaceInputCard
        imageUrl={imageUrl}
        imageFilename={imageFilename}
        cropBox={confirmedCropBox}
        onCropBoxChange={setConfirmedCropBox}
        onImageSelect={handleImageSelect}
        onClear={handleClear}
        onRun={handleRun}
        pipelineState={pipelineState}
        apiResult={apiResult}
      />

      {/* 2. Pipeline Stepper */}
      <PipelineStepper steps={steps} />

      {/* Dynamic Calm Loading Progress */}
      {pipelineState === 'loading' && (
        <DynamicLoadingState
          currentStep={loadingStep}
          elapsedSeconds={elapsedSeconds}
          candidateCount={apiResult?.candidates_evaluated}
        />
      )}

      {/* Search Error State */}
      {pipelineState === 'search_error' && apiResult && (
        <>
          <SearchErrorCard result={apiResult} onRetry={handleRun} />
          <MetricsRow result={apiResult} />
        </>
      )}

      {/* Network / HTTP Error State */}
      {pipelineState === 'error' && errorMessage && (
        <ErrorCard message={errorMessage} />
      )}

      {/* 3. Primary Candidate Result (Verified Match or No Match) */}
      {apiResult && (pipelineState === 'success' || pipelineState === 'failure') && (
        <>
          {apiResult.match_found ? (
            <VerifiedCandidateCard result={apiResult} />
          ) : (
            <UnverifiedCandidateCard result={apiResult} />
          )}

          {/* 4. Compact 4-column Metrics Row */}
          <MetricsRow result={apiResult} />

          {/* 5. Blockchain Proof Card (Rendered only if verified candidate match exists) */}
          {apiResult.match_found && apiResult.blockchain && (
            <BlockchainProofCard blockchain={apiResult.blockchain} />
          )}
        </>
      )}

      {/* 6. Pipeline Activity Log (Collapsible) */}
      <PipelineActivityLog entries={logEntries} />
    </div>
  )
}

// ─── History Page Component ───────────────────────────────────────────────────

function HistoryPage({ historyList }: { historyList: HistoryRecord[] }) {
  return (
    <div className="space-y-4 max-w-3xl mx-auto">
      <div>
        <h1 className="text-xl font-bold text-[#111827] tracking-tight mb-0.5">Verification History</h1>
        <p className="text-xs text-[#667085]">Recorded verification runs during current session.</p>
      </div>

      {historyList.length === 0 ? (
        <Card className="p-8 text-center">
          <div className="w-10 h-10 rounded-full bg-gray-100 flex items-center justify-center mx-auto mb-2">
            <svg width="20" height="24" viewBox="0 0 24 24" fill="none">
              <path d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" stroke="#98A2B3" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>
          <p className="text-xs font-bold text-[#111827] mb-0.5">No verification history yet.</p>
          <p className="text-[11px] text-[#667085]">Run a verification pipeline to record session results.</p>
        </Card>
      ) : (
        <Card className="overflow-hidden">
          <table className="w-full text-xs">
            <thead>
              <tr className="bg-[#F7F8FA] border-b border-[#E5E7EB]">
                {['Date', 'Input File', 'Candidates', 'Similarity', 'Source', 'Blockchain', 'Status'].map(col => (
                  <th key={col} className="text-left text-[10px] font-bold text-[#98A2B3] uppercase tracking-wider px-4 py-2.5">
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {historyList.map((row) => (
                <tr key={row.id} className="border-b border-[#F7F8FA] last:border-0 hover:bg-[#F7F8FA] transition-colors">
                  <td className="px-4 py-3 text-[#667085]">{row.date}</td>
                  <td className="px-4 py-3 font-mono text-[#111827]" title={row.input}>{truncateFilename(row.input, 20)}</td>
                  <td className="px-4 py-3 font-mono text-[#111827]">{row.candidates}</td>
                  <td className="px-4 py-3 font-mono text-[#111827]">{row.similarity}</td>
                  <td className="px-4 py-3 text-[#667085]">{row.source}</td>
                  <td className="px-4 py-3 font-mono font-semibold text-[#059669]">{row.blockchain}</td>
                  <td className="px-4 py-3">
                    <StatusPill status={row.status === 'success' ? 'success' : 'warning'} label={row.status === 'success' ? 'Verified' : 'No match'} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}
    </div>
  )
}

// ─── System Page Component ────────────────────────────────────────────────────

function SystemPage({ healthData, isOnline }: { healthData: SystemHealth | null; isOnline: boolean | null }) {
  const configs = [
    { label: 'Backend API Connection', value: isOnline === true ? 'Online (Connected)' : isOnline === false ? 'Offline (Disconnected)' : 'Checking...', mono: false },
    { label: 'Face Recognition Engine', value: healthData?.face_engine || 'InsightFace · buffalo_l', mono: true },
    { label: 'Embedding Vector Size', value: healthData?.embedding_dimensions ? `${healthData.embedding_dimensions}D L2-Normalized` : '512D L2-Normalized', mono: true },
    { label: 'Search Engine Provider', value: healthData?.serpapi_configured ? 'Google Lens via SerpApi (Configured)' : 'Google Lens via SerpApi', mono: false },
    { label: 'Similarity Algorithm', value: 'L2-normalized cosine similarity', mono: false },
    { label: 'Verification Threshold', value: healthData?.default_threshold ? healthData.default_threshold.toFixed(4) : '0.5000', mono: true },
    { label: 'SHA-256 Fingerprint Engine', value: 'Canonical JSON SHA-256 Hash', mono: true },
    { label: 'Blockchain Ledger', value: healthData?.blockchain || 'Local SHA-256 Hash Chain', mono: false },
  ]

  return (
    <div className="space-y-4 max-w-3xl mx-auto">
      <div>
        <h1 className="text-xl font-bold text-[#111827] tracking-tight mb-0.5">System Configuration</h1>
        <p className="text-xs text-[#667085]">Active pipeline configuration parameters from live API health check.</p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {configs.map(cfg => (
          <Card key={cfg.label} className="p-3.5 hover:border-gray-300 transition-colors">
            <p className="text-[10px] font-bold text-[#98A2B3] uppercase tracking-wider mb-1">{cfg.label}</p>
            <p className={cn('text-xs font-semibold text-[#111827]', cfg.mono && 'font-mono')}>{cfg.value}</p>
          </Card>
        ))}
      </div>

      <Card className="p-3.5 flex items-start gap-2.5 bg-blue-50/60 border-blue-200">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" className="flex-shrink-0 mt-0.5">
          <circle cx="8" cy="8" r="6.5" stroke="#2563EB" strokeWidth="1.2" />
          <path d="M8 7v4M8 5.5v.5" stroke="#2563EB" strokeWidth="1.2" strokeLinecap="round" />
        </svg>
        <p className="text-xs text-blue-900">
          API keys and backend secret configurations remain fully encapsulated on the server and are never transmitted to the frontend.
        </p>
      </Card>
    </div>
  )
}

// ─── Main App Component ───────────────────────────────────────────────────────

export default function App() {
  const [page, setPage] = useState<Page>('verification')
  const [isOnline, setIsOnline] = useState<boolean | null>(null)
  const [healthData, setHealthData] = useState<SystemHealth | null>(null)
  const [historyList, setHistoryList] = useState<HistoryRecord[]>([])

  const checkHealth = useCallback(async () => {
    try {
      const res = await fetch('/api/health')
      if (res.ok) {
        const data: SystemHealth = await res.json()
        setHealthData(data)
        setIsOnline(true)
      } else {
        setIsOnline(false)
      }
    } catch {
      setIsOnline(false)
    }
  }, [])

  useEffect(() => {
    checkHealth()
    const timer = setInterval(checkHealth, 15000)
    return () => clearInterval(timer)
  }, [checkHealth])

  const handleRecordSuccess = (record: HistoryRecord) => {
    setHistoryList(prev => [record, ...prev])
  }

  return (
    <div className="min-h-full bg-[#F7F8FA]">
      <Navbar page={page} setPage={setPage} isOnline={isOnline} />
      <main className="max-w-4xl mx-auto px-4 py-6">
        {page === 'verification' && <VerificationPage onRecordSuccess={handleRecordSuccess} />}
        {page === 'history' && <HistoryPage historyList={historyList} />}
        {page === 'system' && <SystemPage healthData={healthData} isOnline={isOnline} />}
      </main>
    </div>
  )
}
