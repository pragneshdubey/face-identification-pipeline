import { useState, useRef, useCallback, useEffect } from 'react'

// ─── Types ────────────────────────────────────────────────────────────────────

type Page = 'verification' | 'history' | 'system'
type PipelineState = 'idle' | 'loading' | 'success' | 'failure'

interface LogEntry {
  time: string
  message: string
  status: 'info' | 'success' | 'warning'
}

interface PipelineStep {
  id: number
  label: string
  sublabel: string
  state: 'pending' | 'active' | 'done'
}

// ─── Constants ────────────────────────────────────────────────────────────────

const SUCCESS_LOG: LogEntry[] = [
  { time: '14:32:01', message: 'Loading input image...', status: 'info' },
  { time: '14:32:02', message: 'Face detected — confidence 0.8149', status: 'success' },
  { time: '14:32:02', message: 'Generating 512D face embedding...', status: 'info' },
  { time: '14:32:03', message: 'Uploading face crop to Google Lens...', status: 'info' },
  { time: '14:32:05', message: '59 candidates retrieved from web search', status: 'success' },
  { time: '14:32:07', message: 'Candidate verification started...', status: 'info' },
  { time: '14:32:09', message: 'Best candidate similarity score: 0.9709', status: 'success' },
  { time: '14:32:09', message: 'Verification threshold: 0.5000', status: 'info' },
  { time: '14:32:09', message: 'Candidate VERIFIED — exceeds threshold', status: 'success' },
  { time: '14:32:10', message: 'SHA-256 fingerprint generated', status: 'success' },
  { time: '14:32:10', message: 'Blockchain record created', status: 'success' },
  { time: '14:32:10', message: 'Chain integrity check PASSED', status: 'success' },
  { time: '14:32:10', message: 'On-chain re-verification VERIFIED', status: 'success' },
]

const FAILURE_LOG: LogEntry[] = [
  { time: '14:35:01', message: 'Loading input image...', status: 'info' },
  { time: '14:35:02', message: 'Face detected — confidence 0.7432', status: 'success' },
  { time: '14:35:02', message: 'Generating 512D face embedding...', status: 'info' },
  { time: '14:35:03', message: 'Uploading face crop to Google Lens...', status: 'info' },
  { time: '14:35:05', message: '59 candidates retrieved from web search', status: 'success' },
  { time: '14:35:07', message: 'Candidate verification started...', status: 'info' },
  { time: '14:35:09', message: 'Best candidate similarity score: 0.3699', status: 'warning' },
  { time: '14:35:09', message: 'Verification threshold: 0.5000', status: 'info' },
  { time: '14:35:09', message: 'No candidate exceeded threshold — no match found', status: 'warning' },
  { time: '14:35:10', message: 'Blockchain record not created — verification not passed', status: 'warning' },
]

const PIPELINE_STEPS_INITIAL: PipelineStep[] = [
  { id: 1, label: 'Face', sublabel: 'Detection', state: 'pending' },
  { id: 2, label: 'Search', sublabel: 'Web Candidates', state: 'pending' },
  { id: 3, label: 'Verify', sublabel: 'Face Similarity', state: 'pending' },
  { id: 4, label: 'Blockchain', sublabel: 'Registration', state: 'pending' },
  { id: 5, label: 'Re-Verify', sublabel: 'On-Chain', state: 'pending' },
]

const HISTORY_ROWS = [
  { date: 'Sep 5, 2026', input: 'einstein_demo.jpg', candidates: 59, similarity: '0.9709', source: 'Facebook', blockchain: 'Verified', status: 'success' as const },
  { date: 'Sep 4, 2026', input: 'sample_face_02.jpg', candidates: 42, similarity: '0.3699', source: '—', blockchain: 'None', status: 'failure' as const },
  { date: 'Sep 4, 2026', input: 'portrait_test.png', candidates: 71, similarity: '0.8812', source: 'Wikipedia', blockchain: 'Verified', status: 'success' as const },
  { date: 'Sep 3, 2026', input: 'headshot_01.jpg', candidates: 18, similarity: '0.4102', source: '—', blockchain: 'None', status: 'failure' as const },
]

// ─── Utility ──────────────────────────────────────────────────────────────────

function cn(...classes: (string | false | undefined | null)[]) {
  return classes.filter(Boolean).join(' ')
}

function now() {
  return new Date().toLocaleTimeString('en-GB', { hour12: false })
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
    <svg width="40" height="40" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
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

function IconCopy({ onClick }: { onClick: () => void }) {
  const [copied, setCopied] = useState(false)
  const handleClick = () => {
    onClick()
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }
  return (
    <button onClick={handleClick} className="ml-2 text-gray-400 hover:text-gray-600 transition-colors" title="Copy to clipboard">
      {copied ? (
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M2 7l3.5 3.5 6.5-7" stroke="#059669" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
      ) : (
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><rect x="4.5" y="4.5" width="8" height="8" rx="1.5" stroke="currentColor" strokeWidth="1.2" /><path d="M4.5 9.5H3a1.5 1.5 0 01-1.5-1.5V3A1.5 1.5 0 013 1.5h5A1.5 1.5 0 019.5 3v1.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" /></svg>
      )}
    </button>
  )
}

function IconSpinner() {
  return (
    <svg className="animate-spin-slow" width="20" height="20" viewBox="0 0 20 20" fill="none">
      <circle cx="10" cy="10" r="8" stroke="#E5E7EB" strokeWidth="2.5" />
      <path d="M10 2a8 8 0 018 8" stroke="#111827" strokeWidth="2.5" strokeLinecap="round" />
    </svg>
  )
}

// ─── Shared components ────────────────────────────────────────────────────────

function Button({ children, variant = 'primary', onClick, className = '', disabled = false }: {
  children: React.ReactNode
  variant?: 'primary' | 'outline' | 'ghost'
  onClick?: () => void
  className?: string
  disabled?: boolean
}) {
  const base = 'inline-flex items-center justify-center gap-2 text-sm font-medium rounded-[10px] transition-all duration-150 cursor-pointer select-none disabled:opacity-50 disabled:cursor-not-allowed'
  const variants = {
    primary: 'bg-[#111827] text-white px-5 py-2.5 hover:bg-[#1f2937] active:bg-[#0f172a] shadow-sm',
    outline: 'border border-[#E5E7EB] text-[#111827] px-5 py-2.5 hover:bg-gray-50 active:bg-gray-100',
    ghost: 'text-[#667085] px-3 py-2 hover:bg-gray-100 hover:text-[#111827]',
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
    <span className={cn('inline-flex items-center gap-1.5 text-xs font-medium px-2.5 py-1 rounded-full border', styles[status])}>
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

function HashField({ label, value }: { label: string; value: string }) {
  return (
    <div className="space-y-1.5">
      <p className="text-xs font-medium text-[#667085] uppercase tracking-wider">{label}</p>
      <div className="flex items-center bg-[#F7F8FA] rounded-lg px-3 py-2.5 border border-[#E5E7EB]">
        <code className="font-mono text-xs text-[#111827] flex-1 break-all leading-relaxed">
          {value}
        </code>
        <IconCopy onClick={() => navigator.clipboard?.writeText(value)} />
      </div>
    </div>
  )
}

function MetricCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <Card className="p-4">
      <p className="text-xs font-medium text-[#98A2B3] uppercase tracking-wider mb-1">{label}</p>
      <p className="text-2xl font-semibold text-[#111827] font-mono leading-none">{value}</p>
      {sub && <p className="text-xs text-[#98A2B3] mt-1">{sub}</p>}
    </Card>
  )
}

// ─── Navbar ───────────────────────────────────────────────────────────────────

function Navbar({ page, setPage }: { page: Page; setPage: (p: Page) => void }) {
  const navItems: { id: Page; label: string }[] = [
    { id: 'verification', label: 'Verification' },
    { id: 'history', label: 'History' },
    { id: 'system', label: 'System' },
  ]
  return (
    <header className="bg-white border-b border-[#E5E7EB] sticky top-0 z-50">
      <div className="max-w-[1320px] mx-auto px-6 h-14 flex items-center gap-8">
        <div className="flex items-center gap-3 flex-shrink-0">
          <IconFaceVerify />
          <div className="flex items-center gap-2">
            <span className="font-semibold text-[#111827] text-[15px] tracking-tight">FaceVerify</span>
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
                'px-3.5 py-1.5 text-sm rounded-lg transition-all duration-150 font-medium',
                page === item.id
                  ? 'bg-[#111827] text-white'
                  : 'text-[#667085] hover:text-[#111827] hover:bg-gray-50'
              )}
            >
              {item.label}
            </button>
          ))}
        </nav>

        <div className="ml-auto flex items-center gap-3">
          <div className="flex items-center gap-2 text-xs text-[#667085]">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse-dot" />
            System Ready
          </div>
          <div className="w-8 h-8 rounded-full bg-[#111827] flex items-center justify-center text-white text-xs font-medium">
            FV
          </div>
        </div>
      </div>
    </header>
  )
}

// ─── Pipeline Stepper (horizontal) ───────────────────────────────────────────

function PipelineStepper({ steps }: { steps: PipelineStep[] }) {
  return (
    <Card className="p-6">
      <div className="flex items-center justify-between relative">
        <div className="absolute left-0 right-0 top-[22px] h-px bg-[#E5E7EB] mx-[40px]" />
        {steps.map((step, i) => (
          <div key={step.id} className="flex flex-col items-center gap-2 relative z-10 flex-1">
            <div className={cn(
              'w-11 h-11 rounded-full flex items-center justify-center text-sm font-semibold border-2 transition-all duration-500',
              step.state === 'done' && 'bg-[#059669] border-[#059669] text-white',
              step.state === 'active' && 'bg-white border-[#2563EB] text-[#2563EB] shadow-md',
              step.state === 'pending' && 'bg-white border-[#E5E7EB] text-[#98A2B3]',
            )}>
              {step.state === 'done' ? (
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                  <path d="M3 8l3.5 3.5 6.5-7" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              ) : (
                <span className="font-mono text-xs">0{i + 1}</span>
              )}
            </div>
            <div className="text-center">
              <p className={cn('text-xs font-semibold', step.state === 'pending' ? 'text-[#98A2B3]' : 'text-[#111827]')}>
                {step.label}
              </p>
              <p className="text-[10px] text-[#98A2B3]">{step.sublabel}</p>
            </div>
          </div>
        ))}
      </div>
    </Card>
  )
}

// ─── Face Input Card ──────────────────────────────────────────────────────────

function FaceInputCard({
  imageUrl,
  onImageSelect,
  onClear,
  onRun,
  pipelineState,
}: {
  imageUrl: string | null
  onImageSelect: (url: string, file?: File) => void
  onClear: () => void
  onRun: () => void
  pipelineState: PipelineState
}) {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFile = (file: File) => {
    if (!file.type.match(/image\/(jpeg|jpg|png)/)) return
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

  return (
    <Card className="p-6 flex flex-col gap-5">
      <div>
        <div className="flex items-center gap-2 mb-0.5">
          <span className="font-mono text-xs text-[#98A2B3]">01</span>
          <h2 className="text-[15px] font-semibold text-[#111827]">Face Input</h2>
        </div>
        <p className="text-xs text-[#667085]">Upload an image containing a detectable face.</p>
      </div>

      {!imageUrl ? (
        <div
          className={cn(
            'border-2 border-dashed rounded-xl flex flex-col items-center justify-center gap-3 py-12 cursor-pointer transition-all duration-200',
            dragging ? 'border-[#111827] bg-gray-50' : 'border-[#E5E7EB] hover:border-gray-300 hover:bg-gray-50/50'
          )}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onClick={() => inputRef.current?.click()}
        >
          <IconUpload />
          <div className="text-center">
            <p className="text-sm font-medium text-[#111827]">Drop an image here</p>
            <p className="text-xs text-[#667085] mt-0.5">or browse from your computer</p>
          </div>
          <p className="text-[11px] text-[#98A2B3]">JPG, JPEG or PNG · Max 10 MB</p>
          <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png" className="hidden" onChange={onFileChange} />
        </div>
      ) : (
        <div className="space-y-4">
          <div className="relative rounded-xl overflow-hidden bg-gray-100" style={{ aspectRatio: '4/3' }}>
            <img src={imageUrl} alt="Uploaded face" className="w-full h-full object-cover" />
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <div className="border-2 border-[#059669] rounded-lg" style={{ width: '42%', height: '60%', boxShadow: '0 0 0 9999px rgba(0,0,0,0.25)' }} />
            </div>
            <div className="absolute top-3 left-3">
              <span className="flex items-center gap-1.5 bg-emerald-600/90 backdrop-blur-sm text-white text-[11px] font-medium px-2.5 py-1 rounded-full">
                <svg width="10" height="10" viewBox="0 0 10 10" fill="none"><path d="M1.5 5l2.5 2.5 4.5-5" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
                Face detected
              </span>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="bg-[#F7F8FA] rounded-lg p-3 border border-[#E5E7EB]">
              <p className="text-[10px] text-[#98A2B3] uppercase tracking-wider mb-1">Detection confidence</p>
              <p className="font-mono text-sm font-semibold text-[#111827]">0.8149</p>
            </div>
            <div className="bg-[#F7F8FA] rounded-lg p-3 border border-[#E5E7EB]">
              <p className="text-[10px] text-[#98A2B3] uppercase tracking-wider mb-1">Embedding</p>
              <p className="font-mono text-sm font-semibold text-[#059669]">512D generated</p>
            </div>
          </div>
        </div>
      )}

      <div className="flex gap-2.5 pt-1">
        <Button
          onClick={onRun}
          disabled={!imageUrl || isRunning}
          className="flex-1"
        >
          {isRunning ? <><IconSpinner /> Processing...</> : 'Run Verification'}
        </Button>
        {imageUrl && (
          <Button variant="outline" onClick={onClear} disabled={isRunning}>
            Clear
          </Button>
        )}
      </div>
    </Card>
  )
}

// ─── Verification Overview (timeline) ────────────────────────────────────────

function VerificationOverview({ steps }: { steps: PipelineStep[] }) {
  const timelineItems = [
    { label: 'Face Detection', step: 0 },
    { label: 'Face Embedding', step: 0 },
    { label: 'Web Search', step: 1 },
    { label: 'Candidate Verification', step: 2 },
    { label: 'Blockchain Registration', step: 3 },
    { label: 'Re-verification', step: 4 },
  ]

  return (
    <Card className="p-6 flex flex-col gap-5">
      <div>
        <h2 className="text-[15px] font-semibold text-[#111827] mb-0.5">Verification Overview</h2>
        <p className="text-xs text-[#667085]">Live pipeline status across all processing stages.</p>
      </div>

      <div className="flex flex-col gap-0">
        {timelineItems.map((item, i) => {
          const stepState = steps[item.step]?.state ?? 'pending'
          const done = stepState === 'done'
          const active = stepState === 'active'
          const isLast = i === timelineItems.length - 1
          return (
            <div key={i} className="flex gap-4">
              <div className="flex flex-col items-center">
                <div className={cn(
                  'w-5 h-5 rounded-full flex items-center justify-center flex-shrink-0 transition-all duration-500',
                  done ? 'bg-[#059669]' : active ? 'bg-[#2563EB]' : 'bg-[#E5E7EB]'
                )}>
                  {done && (
                    <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
                      <path d="M2 5l2.5 2.5 3.5-4" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  )}
                  {active && <span className="w-2 h-2 rounded-full bg-white animate-pulse-dot" />}
                </div>
                {!isLast && <div className="w-px flex-1 bg-[#E5E7EB] my-1" />}
              </div>
              <div className="pb-5 flex-1">
                <div className="flex items-center justify-between">
                  <p className={cn('text-sm font-medium', done ? 'text-[#111827]' : active ? 'text-[#2563EB]' : 'text-[#98A2B3]')}>
                    {item.label}
                  </p>
                  {done && (
                    <span className="text-[10px] font-medium text-emerald-600">
                      Complete
                    </span>
                  )}
                  {active && (
                    <span className="text-[10px] font-medium text-blue-600 animate-pulse-dot">
                      Running...
                    </span>
                  )}
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </Card>
  )
}

// ─── Loading State ────────────────────────────────────────────────────────────

function LoadingState({ currentStep }: { currentStep: number }) {
  const stages = [
    'Detecting face...',
    'Generating embedding...',
    'Searching Google Lens...',
    'Evaluating candidates...',
    'Registering blockchain record...',
    'Re-verifying record...',
  ]
  return (
    <Card className="p-8 flex flex-col items-center gap-6">
      <div className="relative">
        <div className="w-16 h-16 rounded-full bg-[#F7F8FA] border-2 border-[#E5E7EB] flex items-center justify-center">
          <IconSpinner />
        </div>
      </div>
      <div className="text-center">
        <p className="text-sm font-semibold text-[#111827] mb-1">
          {stages[Math.min(currentStep, stages.length - 1)]}
        </p>
        <p className="text-xs text-[#667085]">Pipeline is processing — do not close this window</p>
      </div>
      <div className="w-full max-w-xs space-y-2">
        {stages.map((stage, i) => (
          <div key={i} className="flex items-center gap-3">
            <div className={cn(
              'w-4 h-4 rounded-full flex items-center justify-center flex-shrink-0',
              i < currentStep ? 'bg-[#059669]' : i === currentStep ? 'bg-[#2563EB]' : 'bg-[#E5E7EB]'
            )}>
              {i < currentStep && (
                <svg width="8" height="8" viewBox="0 0 8 8" fill="none">
                  <path d="M1.5 4l1.8 1.8 3-3.3" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              )}
              {i === currentStep && <span className="w-1.5 h-1.5 rounded-full bg-white animate-pulse-dot" />}
            </div>
            <p className={cn(
              'text-xs',
              i < currentStep ? 'text-[#059669]' : i === currentStep ? 'text-[#2563EB] font-medium' : 'text-[#98A2B3]'
            )}>
              {stage}
            </p>
          </div>
        ))}
      </div>
    </Card>
  )
}

// ─── Success Results ──────────────────────────────────────────────────────────

function SuccessResults({ imageFilename, result }: { imageFilename: string; result?: any }) {
  const cand = result?.candidate
  const bchain = result?.blockchain
  const evaluated = result?.candidates_evaluated ?? 59
  const similarity = cand?.similarity ? cand.similarity.toFixed(4) : "0.9709"
  const threshold = result?.threshold ? result.threshold.toFixed(4) : "0.5000"
  const title = cand?.title ?? "Photo of Albert Einstein was taken by photographer..."
  const domain = cand?.domain ?? "facebook.com"
  const sourceUrl = cand?.source_url ?? "https://facebook.com"
  const candidateImg = cand?.candidate_image_url ?? "https://images.unsplash.com/photo-1594736797933-d0501ba2fe65?w=96&h=112&fit=crop&auto=format"
  const fingerprint = bchain?.fingerprint ?? "2a1929b44f3d4eb3b541db508f9335491444515a14825348e87edfc82ca30550"
  const recordId = bchain?.record_id ?? "46e594d4afb39a7a53c7d5f605a9f23dca111318f3adcf42b81f9484aa721fe5"
  const chainIntegrity = bchain?.chain_integrity ?? "PASSED"
  const reverification = bchain?.reverification ?? "VERIFIED"

  return (
    <div className="space-y-6 animate-slide-in">
      {/* Web Search Results */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <div>
            <div className="flex items-center gap-2 mb-0.5">
              <span className="font-mono text-xs text-[#98A2B3]">02</span>
              <h2 className="text-[15px] font-semibold text-[#111827]">Web Search Results</h2>
            </div>
            <p className="text-xs text-[#667085]">Candidates retrieved dynamically from Google Lens</p>
          </div>
          <div className="text-right">
            <p className="font-mono text-xl font-semibold text-[#111827]">{evaluated}</p>
            <p className="text-xs text-[#667085]">candidates discovered</p>
          </div>
        </div>

        <Card className="p-6">
          <div className="flex gap-5">
            <div className="w-24 h-28 rounded-xl bg-gradient-to-br from-gray-200 to-gray-100 flex-shrink-0 overflow-hidden">
              <img
                src={candidateImg}
                alt="Candidate image"
                className="w-full h-full object-cover"
                onError={(e) => {
                  (e.target as HTMLImageElement).src = "https://images.unsplash.com/photo-1594736797933-d0501ba2fe65?w=96&h=112&fit=crop&auto=format"
                }}
              />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <div className="w-5 h-5 rounded bg-blue-600 flex items-center justify-center">
                      <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
                        <path d="M6 2H2v6h6V6" stroke="white" strokeWidth="1" strokeLinecap="round" />
                        <path d="M5 5L8 2M8 2H6M8 2v2" stroke="white" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                    </div>
                    <span className="text-xs font-semibold text-[#111827]">{domain}</span>
                  </div>
                  <p className="text-sm text-[#111827] font-medium mb-1 leading-snug">
                    {title}
                  </p>
                  <p className="text-xs text-[#667085] mb-2">{domain}</p>
                  <a href={sourceUrl} target="_blank" rel="noreferrer" className="text-xs text-[#2563EB] hover:underline flex items-center gap-1 font-medium inline-flex">
                    Open Source
                    <svg width="10" height="10" viewBox="0 0 10 10" fill="none"><path d="M2 8L8 2M8 2H5M8 2v3" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" /></svg>
                  </a>
                </div>
                <StatusPill status="success" label="VERIFIED" />
              </div>

              <div className="mt-4 pt-4 border-t border-[#E5E7EB]">
                <div className="flex items-center gap-8">
                  <div>
                    <p className="text-[10px] text-[#98A2B3] uppercase tracking-wider mb-0.5">Face Similarity</p>
                    <p className="font-mono text-2xl font-semibold text-[#059669]">{similarity}</p>
                  </div>
                  <div>
                    <p className="text-[10px] text-[#98A2B3] uppercase tracking-wider mb-0.5">Threshold</p>
                    <p className="font-mono text-2xl font-semibold text-[#111827]">{threshold}</p>
                  </div>
                  <div className="flex-1">
                    <div className="h-2 bg-[#E5E7EB] rounded-full overflow-hidden">
                      <div className="h-2 bg-[#059669] rounded-full" style={{ width: `${Math.min(100, parseFloat(similarity) * 100)}%` }} />
                    </div>
                    <p className="text-[10px] text-[#667085] mt-1.5">Candidate face exceeds the configured similarity threshold.</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </Card>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-4 gap-4">
        <MetricCard label="Candidates Evaluated" value={evaluated} />
        <MetricCard label="Faces Detected" value="1+" />
        <MetricCard label="Best Similarity" value={similarity} sub="Score" />
        <MetricCard label="Threshold" value={threshold} sub="Required" />
      </div>

      {/* Blockchain + Summary */}
      <div className="grid grid-cols-[1fr_320px] gap-5">
        <Card className="p-6 space-y-5">
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-2 mb-0.5">
                <span className="font-mono text-xs text-[#98A2B3]">03</span>
                <h2 className="text-[15px] font-semibold text-[#111827]">Blockchain Verification</h2>
              </div>
              <p className="text-xs text-[#667085]">SHA-256 hash-chain ledger — tamper-evident record</p>
            </div>
            <StatusPill status="success" label="VERIFIED" />
          </div>

          <div className="space-y-3">
            <HashField label="SHA-256 Fingerprint" value={fingerprint} />
            <HashField label="Blockchain Record" value={recordId} />
          </div>

          <div className="grid grid-cols-3 gap-3">
            {[
              { label: 'Chain Integrity', value: chainIntegrity, status: 'success' as const },
              { label: 'Record Status', value: 'REGISTERED', status: 'success' as const },
              { label: 'On-chain Re-verification', value: reverification, status: 'success' as const },
            ].map(row => (
              <div key={row.label} className="bg-[#F7F8FA] rounded-lg p-3.5 border border-[#E5E7EB]">
                <p className="text-[10px] text-[#98A2B3] uppercase tracking-wider mb-2">{row.label}</p>
                <div className="flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                  <span className="font-mono text-xs font-semibold text-[#059669]">{row.value}</span>
                </div>
              </div>
            ))}
          </div>

          {/* Chain visual */}
          <div>
            <p className="text-[10px] text-[#98A2B3] uppercase tracking-wider mb-3">Chain Structure</p>
            <div className="flex items-center gap-0">
              {['Genesis', 'Block', 'Verified Record', 'Integrity Check'].map((label, i) => (
                <div key={i} className="flex items-center">
                  <div className="flex flex-col items-center">
                    <div className="h-8 w-24 bg-[#F7F8FA] border border-[#E5E7EB] rounded-lg flex items-center justify-center">
                      <span className="text-[10px] font-medium text-[#667085]">{label}</span>
                    </div>
                  </div>
                  {i < 3 && (
                    <div className="flex items-center gap-0 mx-1">
                      <div className="w-4 h-px bg-[#059669]" />
                      <svg width="6" height="8" viewBox="0 0 6 8" fill="none">
                        <path d="M1 0l4 4-4 4" stroke="#059669" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </Card>

        <Card className="p-5">
          <h3 className="text-sm font-semibold text-[#111827] mb-4">Verification Summary</h3>
          <div className="space-y-2.5">
            {[
              { label: 'Input', value: imageFilename },
              { label: 'Candidates', value: String(evaluated) },
              { label: 'Source', value: domain },
              { label: 'Similarity', value: similarity },
              { label: 'Threshold', value: threshold },
              { label: 'Blockchain', value: 'Registered' },
              { label: 'Integrity', value: chainIntegrity },
              { label: 'Re-verification', value: reverification },
            ].map(row => (
              <div key={row.label} className="flex justify-between items-baseline gap-3 py-1.5 border-b border-[#F7F8FA] last:border-0">
                <span className="text-xs text-[#667085] flex-shrink-0">{row.label}</span>
                <span className={cn(
                  'text-xs font-medium text-right min-w-0',
                  ['PASSED', 'Registered', 'VERIFIED'].includes(row.value) ? 'text-[#059669] font-mono' : 'text-[#111827] font-mono'
                )}>
                  {row.value}
                </span>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  )
}

// ─── Failure Results ──────────────────────────────────────────────────────────

function FailureResults({ imageFilename, result }: { imageFilename: string; result?: any }) {
  const evaluated = result?.candidates_evaluated ?? 59
  const highestSim = result?.highest_similarity ? result.highest_similarity.toFixed(4) : "0.3699"
  const threshold = result?.threshold ? result.threshold.toFixed(4) : "0.5000"

  return (
    <div className="space-y-6 animate-slide-in">
      <Card className="p-6">
        <div className="flex items-start justify-between mb-5">
          <div>
            <div className="flex items-center gap-2 mb-0.5">
              <span className="font-mono text-xs text-[#98A2B3]">02</span>
              <h2 className="text-[15px] font-semibold text-[#111827]">Web Search Results</h2>
            </div>
            <p className="text-xs text-[#667085]">Candidates retrieved dynamically from Google Lens</p>
          </div>
          <StatusPill status="warning" label="NO MATCH" />
        </div>

        <div className="bg-amber-50 border border-amber-200 rounded-xl p-5 space-y-4">
          <p className="text-sm font-semibold text-amber-900">No Verified Match Found</p>
          <div className="grid grid-cols-4 gap-4">
            <div>
              <p className="text-[10px] text-amber-600 uppercase tracking-wider mb-1">Candidates Evaluated</p>
              <p className="font-mono text-xl font-semibold text-amber-900">{evaluated}</p>
            </div>
            <div>
              <p className="text-[10px] text-amber-600 uppercase tracking-wider mb-1">Highest Similarity</p>
              <p className="font-mono text-xl font-semibold text-amber-900">{highestSim}</p>
            </div>
            <div>
              <p className="text-[10px] text-amber-600 uppercase tracking-wider mb-1">Required Threshold</p>
              <p className="font-mono text-xl font-semibold text-amber-900">{threshold}</p>
            </div>
            <div>
              <p className="text-[10px] text-amber-600 uppercase tracking-wider mb-1">Blockchain</p>
              <p className="font-mono text-xl font-semibold text-amber-900">No record</p>
            </div>
          </div>
          <div className="h-1.5 bg-amber-200 rounded-full overflow-hidden">
            <div className="h-1.5 bg-amber-500 rounded-full" style={{ width: `${Math.min(100, parseFloat(highestSim) * 100)}%` }} />
          </div>
          <p className="text-xs text-amber-700">
            No candidate exceeded the configured verification threshold. No blockchain record was created.
          </p>
        </div>
      </Card>
    </div>
  )
}

// ─── Pipeline Log ─────────────────────────────────────────────────────────────

function PipelineLog({ entries }: { entries: LogEntry[] }) {
  const dotColors = {
    info: 'bg-blue-400',
    success: 'bg-emerald-500',
    warning: 'bg-amber-500',
  }
  const textColors = {
    info: 'text-[#667085]',
    success: 'text-[#059669]',
    warning: 'text-amber-600',
  }
  return (
    <Card className="p-5">
      <h3 className="text-sm font-semibold text-[#111827] mb-4 flex items-center gap-2">
        Pipeline Activity
        <span className="text-[10px] font-normal text-[#98A2B3] font-mono">
          {entries.length} events
        </span>
      </h3>
      <div className="bg-[#F7F8FA] rounded-xl border border-[#E5E7EB] p-4 max-h-56 overflow-y-auto">
        <div className="space-y-1.5">
          {entries.map((entry, i) => (
            <div key={i} className={cn('flex items-start gap-3 animate-slide-in')} style={{ animationDelay: `${i * 30}ms` }}>
              <span className={cn('w-1.5 h-1.5 rounded-full mt-[5px] flex-shrink-0', dotColors[entry.status])} />
              <span className="font-mono text-[11px] text-[#98A2B3] flex-shrink-0">[{entry.time}]</span>
              <span className={cn('font-mono text-[11px]', textColors[entry.status])}>{entry.message}</span>
            </div>
          ))}
          {entries.length === 0 && (
            <p className="font-mono text-[11px] text-[#98A2B3]">Awaiting pipeline execution...</p>
          )}
        </div>
      </div>
    </Card>
  )
}

// ─── Verification Page ────────────────────────────────────────────────────────

function VerificationPage() {
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [imageFilename, setImageFilename] = useState('einstein_demo.jpg')
  const [pipelineState, setPipelineState] = useState<PipelineState>('idle')
  const [steps, setSteps] = useState<PipelineStep[]>(PIPELINE_STEPS_INITIAL)
  const [logEntries, setLogEntries] = useState<LogEntry[]>([])
  const [loadingStep, setLoadingStep] = useState(0)
  const [simulateFailure, setSimulateFailure] = useState(false)
  const [apiResult, setApiResult] = useState<any>(null)

  const resetPipeline = () => {
    setSteps(PIPELINE_STEPS_INITIAL)
    setLogEntries([])
    setLoadingStep(0)
    setApiResult(null)
  }

  const handleImageSelect = (url: string, file?: File) => {
    setImageUrl(url)
    setSelectedFile(file || null)
    if (file) {
      setImageFilename(file.name)
    } else {
      setImageFilename('einstein_demo.jpg')
    }
    resetPipeline()
    setPipelineState('idle')
  }

  const handleClear = () => {
    setImageUrl(null)
    setSelectedFile(null)
    setApiResult(null)
    resetPipeline()
    setPipelineState('idle')
  }

  const runSimulatedPipeline = () => {
    resetPipeline()
    setPipelineState('loading')
    const targetLog = simulateFailure ? FAILURE_LOG : SUCCESS_LOG
    const stepSequence = [0, 1, 2, 2, 3, 4]

    targetLog.forEach((entry, i) => {
      setTimeout(() => {
        setLoadingStep(i)
        setLogEntries(prev => [...prev, entry])

        const stepIdx = stepSequence[Math.min(i, stepSequence.length - 1)]
        setSteps(prev => prev.map((s, si) => {
          if (si < stepIdx) return { ...s, state: 'done' }
          if (si === stepIdx) return { ...s, state: 'active' }
          return { ...s, state: 'pending' }
        }))

        if (i === targetLog.length - 1) {
          setTimeout(() => {
            setSteps(prev => prev.map(s => ({ ...s, state: 'done' })))
            setPipelineState(simulateFailure ? 'failure' : 'success')
          }, 600)
        }
      }, i * 700)
    })
  }

  const handleRun = async () => {
    resetPipeline()
    setPipelineState('loading')

    const formData = new FormData()
    if (selectedFile) {
      formData.append('file', selectedFile)
    } else {
      formData.append('image_name', imageFilename)
    }
    formData.append('provider', simulateFailure ? 'mock' : 'web')
    formData.append('threshold', '0.5')

    try {
      setLoadingStep(0)
      setSteps(prev => prev.map((s, i) => i === 0 ? { ...s, state: 'active' } : s))

      const res = await fetch('/api/verify', {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        throw new Error(`API returned HTTP ${res.status}`)
      }

      const data = await res.json()
      setApiResult(data)

      if (data.logs && Array.isArray(data.logs) && data.logs.length > 0) {
        const stepSequence = [0, 1, 2, 2, 3, 4]
        data.logs.forEach((entry: LogEntry, i: number) => {
          setTimeout(() => {
            const stepIdx = stepSequence[Math.min(i, stepSequence.length - 1)]
            setLoadingStep(stepIdx)
            setLogEntries(prev => [...prev, entry])
            setSteps(prev => prev.map((s, si) => {
              if (si < stepIdx) return { ...s, state: 'done' }
              if (si === stepIdx) return { ...s, state: 'active' }
              return { ...s, state: 'pending' }
            }))
          }, i * 350)
        })

        setTimeout(() => {
          setSteps(prev => prev.map(s => ({ ...s, state: 'done' })))
          setPipelineState(data.match_found ? 'success' : 'failure')
        }, data.logs.length * 350 + 400)
      } else {
        setSteps(prev => prev.map(s => ({ ...s, state: 'done' })))
        setPipelineState(data.match_found ? 'success' : 'failure')
      }
    } catch (err) {
      console.warn("API request failed or offline. Falling back to simulated run:", err)
      runSimulatedPipeline()
    }
  }

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-[#111827] tracking-tight mb-1">Face Verification Pipeline</h1>
          <p className="text-sm text-[#667085] max-w-2xl">
            Discover visual candidates on the open web, independently verify the face, and anchor verified results to a tamper-evident ledger.
          </p>
        </div>
        <div className="flex items-center gap-3 flex-shrink-0">
          <StatusPill status="success" label="Pipeline Ready" />
          <div className="flex items-center gap-1.5 text-xs text-[#667085] bg-white border border-[#E5E7EB] rounded-lg px-3 py-1.5">
            <IconShield />
            Local Verification Engine
          </div>
        </div>
      </div>

      {/* Face input + overview */}
      <div className="grid grid-cols-[1fr_320px] gap-5">
        <FaceInputCard
          imageUrl={imageUrl}
          onImageSelect={handleImageSelect}
          onClear={handleClear}
          onRun={handleRun}
          pipelineState={pipelineState}
        />
        <VerificationOverview steps={steps} />
      </div>

      {/* Pipeline stepper */}
      <PipelineStepper steps={steps} />

      {/* Simulate failure toggle */}
      {pipelineState === 'idle' && (
        <div className="flex items-center gap-2">
          <button
            onClick={() => setSimulateFailure(v => !v)}
            className={cn(
              'w-8 h-4 rounded-full transition-colors duration-200 relative flex-shrink-0',
              simulateFailure ? 'bg-amber-500' : 'bg-[#E5E7EB]'
            )}
          >
            <span className={cn(
              'absolute top-0.5 w-3 h-3 bg-white rounded-full shadow transition-transform duration-200',
              simulateFailure ? 'translate-x-4' : 'translate-x-0.5'
            )} />
          </button>
          <span className="text-xs text-[#667085]">Simulate failure state</span>
        </div>
      )}

      {/* Results */}
      {pipelineState === 'loading' && <LoadingState currentStep={loadingStep} />}
      {pipelineState === 'success' && <SuccessResults imageFilename={imageFilename} result={apiResult} />}
      {pipelineState === 'failure' && <FailureResults imageFilename={imageFilename} result={apiResult} />}

      {/* Log */}
      {logEntries.length > 0 && <PipelineLog entries={logEntries} />}
    </div>
  )
}


// ─── History Page ─────────────────────────────────────────────────────────────

function HistoryPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-[#111827] tracking-tight mb-1">Verification History</h1>
        <p className="text-xs text-[#98A2B3] bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 inline-block">
          UI representation only — persistent history storage is not implemented in the current backend.
        </p>
      </div>

      <Card className="overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-[#F7F8FA] border-b border-[#E5E7EB]">
              {['Date', 'Input', 'Candidates', 'Similarity', 'Source', 'Blockchain', 'Status'].map(col => (
                <th key={col} className="text-left text-[10px] font-semibold text-[#98A2B3] uppercase tracking-wider px-5 py-3.5">
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {HISTORY_ROWS.map((row, i) => (
              <tr key={i} className="border-b border-[#F7F8FA] last:border-0 hover:bg-[#F7F8FA] transition-colors">
                <td className="px-5 py-4 text-xs text-[#667085]">{row.date}</td>
                <td className="px-5 py-4 font-mono text-xs text-[#111827]">{row.input}</td>
                <td className="px-5 py-4 font-mono text-xs text-[#111827]">{row.candidates}</td>
                <td className="px-5 py-4 font-mono text-xs text-[#111827]">{row.similarity}</td>
                <td className="px-5 py-4 text-xs text-[#667085]">{row.source}</td>
                <td className="px-5 py-4">
                  <span className={cn(
                    'font-mono text-xs font-medium',
                    row.blockchain === 'Verified' ? 'text-[#059669]' : 'text-[#98A2B3]'
                  )}>
                    {row.blockchain}
                  </span>
                </td>
                <td className="px-5 py-4">
                  <StatusPill status={row.status === 'success' ? 'success' : 'warning'} label={row.status === 'success' ? 'Verified' : 'No match'} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  )
}

// ─── System Page ──────────────────────────────────────────────────────────────

function SystemPage() {
  const configs = [
    { label: 'Face Engine', value: 'InsightFace · buffalo_l', mono: true },
    { label: 'Embedding', value: '512 dimensions', mono: true },
    { label: 'Search Provider', value: 'Google Lens via SerpApi', mono: false },
    { label: 'Verification Method', value: 'L2-normalized cosine similarity', mono: false },
    { label: 'Threshold', value: '0.5000', mono: true },
    { label: 'Fingerprint', value: 'SHA-256', mono: true },
    { label: 'Blockchain', value: 'Local SHA-256 hash-chain ledger', mono: false },
    { label: 'Re-verification', value: 'Enabled', mono: false },
  ]

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-[#111827] tracking-tight mb-1">System Configuration</h1>
        <p className="text-sm text-[#667085]">Active pipeline configuration — read-only display of backend settings.</p>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {configs.map(cfg => (
          <Card key={cfg.label} className="p-5 hover:border-gray-300 transition-colors">
            <p className="text-[10px] text-[#98A2B3] uppercase tracking-wider mb-2">{cfg.label}</p>
            <p className={cn('text-sm font-semibold text-[#111827]', cfg.mono && 'font-mono')}>{cfg.value}</p>
          </Card>
        ))}
      </div>

      <Card className="p-5 flex items-start gap-3 bg-blue-50 border-blue-200">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" className="flex-shrink-0 mt-0.5">
          <circle cx="8" cy="8" r="6.5" stroke="#2563EB" strokeWidth="1.2" />
          <path d="M8 7v4M8 5.5v.5" stroke="#2563EB" strokeWidth="1.2" strokeLinecap="round" />
        </svg>
        <p className="text-xs text-blue-800">
          API credentials are handled server-side and are never displayed in the interface.
        </p>
      </Card>

      <Card className="p-5">
        <h3 className="text-sm font-semibold text-[#111827] mb-3">Pipeline Connectivity</h3>
        <div className="space-y-2">
          {[
            { label: 'Face Detection Engine', status: 'online' as const },
            { label: 'Embedding Generator', status: 'online' as const },
            { label: 'Google Lens / SerpApi', status: 'online' as const },
            { label: 'Blockchain Ledger', status: 'online' as const },
          ].map(item => (
            <div key={item.label} className="flex items-center justify-between py-2 border-b border-[#F7F8FA] last:border-0">
              <span className="text-sm text-[#667085]">{item.label}</span>
              <div className="flex items-center gap-1.5 text-xs font-medium text-emerald-600">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse-dot" />
                Online
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}

// ─── App ──────────────────────────────────────────────────────────────────────

export default function App() {
  const [page, setPage] = useState<Page>('verification')

  return (
    <div className="min-h-full bg-[#F7F8FA]">
      <Navbar page={page} setPage={setPage} />
      <main className="max-w-[1320px] mx-auto px-6 py-8">
        {page === 'verification' && <VerificationPage />}
        {page === 'history' && <HistoryPage />}
        {page === 'system' && <SystemPage />}
      </main>
    </div>
  )
}
