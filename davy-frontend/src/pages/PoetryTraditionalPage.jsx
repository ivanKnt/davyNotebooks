import { useEffect, useState } from 'react'
import { runPoetry, listPoetryNotebooks, listPoetryPages, listPoetryPagesForNotebook } from '../services/api.js'

export default function PoetryTraditionalPage() {
  const [books, setBooks] = useState([])
  const [pages, setPages] = useState([])
  const [selected, setSelected] = useState('')
  const [running, setRunning] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')
  const [minConfidence, setMinConfidence] = useState(0)

  const load = async () => {
    const [b, p] = await Promise.all([listPoetryNotebooks(), listPoetryPages()])
    setBooks(b.data); setPages(p.data)
  }
  useEffect(() => { load() }, [])

  const onRun = async () => {
    setRunning(true)
    try { await runPoetry(); await load() } finally { setRunning(false) }
  }

  const filterByNotebook = async (id) => {
    setSelected(id)
    if (!id) { const p = await listPoetryPages(); setPages(p.data); return }
    const res = await listPoetryPagesForNotebook(id)
    setPages(res.data)
  }

  const formatPct = (v) => {
    const num = typeof v === 'string' ? parseFloat(v) : (v ?? 0)
    if (!isFinite(num)) return '0%'
    return `${(num * 100).toFixed(1)}%`
  }

  // Filter pages based on search and confidence
  const filteredPages = pages.filter(p => {
    const matchesSearch = !searchTerm || 
      p.notebook_id?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      p.page_consensus?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      String(p.page_number).includes(searchTerm)
    
    const poetryPct = typeof p.poetry_percentage === 'string' ? parseFloat(p.poetry_percentage) : (p.poetry_percentage ?? 0)
    const matchesConfidence = poetryPct >= minConfidence
    
    return matchesSearch && matchesConfidence
  })

  return (
    <div className="bg-gradient-to-b from-white to-slate-100 min-h-screen">
      <div className="max-w-6xl mx-auto px-4 py-12 space-y-8">
        <header className="space-y-3 text-center sm:text-left">
          <h1 className="text-3xl font-semibold text-slate-900">Poetry Explorer (Traditional)</h1>
          <p className="text-slate-600 text-base sm:max-w-3xl">
            Discover pages classified as poetry using citizen science classifications. Filter by notebook and explore poetic content confidence scores.
          </p>
        </header>

        <div className="rounded-3xl border border-slate-200 bg-white/90 shadow-sm p-6 space-y-4">
          <div className="flex flex-wrap items-center gap-3">
            <button 
              onClick={onRun} 
              disabled={running} 
              className="inline-flex items-center gap-2 rounded-full bg-violet-600 px-5 py-2.5 text-sm font-medium text-white shadow-sm transition hover:bg-violet-500 disabled:cursor-not-allowed disabled:bg-violet-300"
            >
              {running ? 'Processing Poetry Classifications…' : 'Run Poetry Classification'}
            </button>
            <select 
              value={selected} 
              onChange={(e)=>filterByNotebook(e.target.value)} 
              className="flex-1 min-w-[240px] rounded-full border border-slate-200 bg-white px-4 py-2.5 text-sm text-slate-700 shadow-sm transition focus:border-violet-300 focus:outline-none focus:ring-2 focus:ring-violet-100"
            >
              <option value="">All notebooks…</option>
              {books.map(b => <option key={b.notebook_id} value={b.notebook_id}>{b.notebook_id} — {b.notebook_title}</option>)}
            </select>
          </div>
          
          <div className="grid gap-3 sm:grid-cols-[1fr_auto_auto]">
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Search by notebook, page, or consensus..."
              className="rounded-full border border-slate-200 bg-white px-4 py-2 text-sm text-slate-700 shadow-sm transition focus:border-violet-300 focus:outline-none focus:ring-2 focus:ring-violet-100"
            />
            <div className="flex items-center gap-2">
              <label className="text-xs text-slate-600 font-medium whitespace-nowrap">Min confidence:</label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={minConfidence}
                onChange={(e) => setMinConfidence(parseFloat(e.target.value))}
                className="w-32"
              />
              <span className="text-xs font-medium text-violet-600 w-12">{(minConfidence * 100).toFixed(0)}%</span>
            </div>
            <span className="text-sm text-slate-500 flex items-center whitespace-nowrap">
              {filteredPages.length} of {pages.length} shown
            </span>
          </div>
        </div>

        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {filteredPages.slice(0, 60).map((p, idx) => (
            <div key={idx} className="group rounded-3xl border border-violet-100 bg-white/90 p-5 shadow-sm transition hover:shadow-md hover:border-violet-200">
              <div className="flex items-start justify-between mb-3">
                <div>
                  <div className="text-sm font-medium text-violet-600">{p.notebook_id}</div>
                  <div className="text-xs text-slate-500">Page {p.page_number}</div>
                </div>
                <span className="inline-flex items-center rounded-full bg-violet-100 px-2.5 py-1 text-xs font-medium text-violet-700">
                  {formatPct(p.poetry_percentage)}
                </span>
              </div>
              <div className="mb-3">
                <div className="text-sm font-semibold text-slate-900">{p.page_consensus}</div>
              </div>
              {p.all_classifications && (
                <div className="space-y-1.5 text-xs">
                  {Object.entries(p.all_classifications)
                    .sort((a,b) => (b[1] ?? 0) - (a[1] ?? 0))
                    .map(([label, val]) => (
                      <div key={label} className="flex items-center justify-between">
                        <span className={label === p.page_consensus ? 'font-semibold text-slate-900' : 'text-slate-600'}>{label}</span>
                        <span className={label === p.page_consensus ? 'font-semibold text-violet-600' : 'text-slate-500'}>{formatPct(val)}</span>
                      </div>
                    ))}
                </div>
              )}
            </div>
          ))}
        </div>

        {filteredPages.length === 0 && pages.length > 0 && (
          <div className="rounded-3xl border border-amber-200 bg-amber-50 p-12 text-center shadow-sm">
            <p className="text-amber-900 font-medium">No pages match your filters.</p>
            <p className="text-sm text-amber-700 mt-2">Try adjusting your search term or confidence threshold.</p>
          </div>
        )}

        {pages.length === 0 && (
          <div className="rounded-3xl border border-slate-200 bg-white/90 p-12 text-center shadow-sm">
            <p className="text-slate-600">No poetry pages found. Run poetry classification or select a different notebook.</p>
          </div>
        )}
      </div>
    </div>
  )
}


