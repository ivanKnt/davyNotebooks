import { useEffect, useState } from 'react'
import { listTRNotebooks, listTRConfigs, runTextReuse, getTextReuseResults } from '../services/api.js'

const badgeClasses = {
  ngram: 'bg-indigo-100 text-indigo-700',
  gst: 'bg-emerald-100 text-emerald-700',
  tfidf: 'bg-amber-100 text-amber-700'
}

export default function TextReuseTraditionalPage() {
  const [notebooks, setNotebooks] = useState([])
  const [alg, setAlg] = useState('ngram')
  const [configs, setConfigs] = useState([])
  const [configId, setConfigId] = useState('')
  const [nb1, setNb1] = useState('')
  const [nb2, setNb2] = useState('')
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [idx, setIdx] = useState(0)
  const [cacheStatus, setCacheStatus] = useState('idle') // idle | fetch | run
  const [error, setError] = useState(null)

  const load = async () => {
    const nbs = await listTRNotebooks(); setNotebooks(nbs.data)
  }
  useEffect(() => { load() }, [])

  useEffect(() => {
    const go = async () => {
      const res = await listTRConfigs(alg)
      setConfigs(res.data); setConfigId(res.data[0]?.config_id || '')
    }
    go()
  }, [alg])

  // Keyboard navigation for instances
  useEffect(() => {
    const handleKeyPress = (e) => {
      if (!results || instances.length === 0) return
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return
      
      if (e.key === 'ArrowLeft' && idx > 0) {
        setIdx(i => Math.max(0, i - 1))
        e.preventDefault()
      } else if (e.key === 'ArrowRight' && idx < instances.length - 1) {
        setIdx(i => Math.min(instances.length - 1, i + 1))
        e.preventDefault()
      }
    }
    window.addEventListener('keydown', handleKeyPress)
    return () => window.removeEventListener('keydown', handleKeyPress)
  }, [results, idx])

  const fetchOrRun = async () => {
    if (!nb1 || !nb2 || !configId) return
    setLoading(true)
    setResults(null)
    setError(null)
    setCacheStatus('fetch')
    try {
      let existingResponse = null
      try {
        existingResponse = await getTextReuseResults(alg, configId, `${nb1},${nb2}`)
      } catch (err) {
        existingResponse = null
      }

      if (existingResponse && existingResponse.status === 200 && (existingResponse.data?.status === 'ok' || existingResponse.data?.status === 'already_exists')) {
        setResults(existingResponse.data.data)
        setIdx(0)
        setCacheStatus('idle')
        return
      }

      setCacheStatus('run')
      const runResponse = await runTextReuse({ algorithm: alg, config_id: Number(configId), notebooks: [nb1, nb2], filename: 'page_to_text.json' })
      const status = runResponse.data?.status
      if (status === 'success' || status === 'already_exists') {
        if (runResponse.data?.data) {
          setResults(runResponse.data.data)
          setIdx(0)
          setCacheStatus('idle')
          return
        }

        const refreshed = await getTextReuseResults(alg, configId, `${nb1},${nb2}`)
        if (refreshed.status === 200 && refreshed.data?.data) {
          setResults(refreshed.data.data)
          setIdx(0)
          setCacheStatus('idle')
          return
        }
      }

      if (status === 'partial') {
        setError(runResponse.data?.message || 'Run completed but results file not located yet')
        setCacheStatus('idle')
        return
      }

      const fallbackMessage = runResponse.data?.message || 'Unable to retrieve text reuse results.'
      setError(fallbackMessage)
      setCacheStatus('idle')
    } catch (err) {
      const message = err?.response?.data?.message || err?.message || 'Unexpected error while running text reuse.'
      setError(message)
      setCacheStatus('idle')
    } finally {
      setLoading(false)
    }
  }

  const instances = results?.reuse_instances || []
  const current = instances[idx] || null
  const score = current ? (
    alg === 'ngram' ? (current.jaccard ?? current.max_containment) :
    alg === 'gst' ? current.gst_similarity :
    current.tfidf_similarity
  ) : null
  const fmt = (v, decimals = 3) => (v === null || v === undefined)
    ? '-'
    : (typeof v === 'number' ? v.toFixed(decimals) : String(v))
  const renderMetadata = (meta, label) => {
    if (!meta || Object.keys(meta).length === 0) return null
    return (
      <details className="mt-3">
        <summary className="cursor-pointer text-xs text-indigo-600 hover:text-indigo-700 font-medium transition">{label}</summary>
        <div className="mt-3 grid gap-2 sm:grid-cols-2 text-xs">
          {Object.entries(meta).map(([key, value]) => (
            <div key={key} className="rounded-xl bg-slate-50 p-3 border border-slate-100">
              <div className="text-slate-500 font-medium mb-1">{key.replace(/_/g, ' ')}</div>
              <div className="text-slate-900 font-mono text-[11px] break-words">
                {typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)}
              </div>
            </div>
          ))}
        </div>
      </details>
    )
  }

  return (
    <div className="bg-gradient-to-b from-white to-slate-100 min-h-screen">
      <div className="max-w-6xl mx-auto px-4 py-12 space-y-8">
        <header className="space-y-3 text-center sm:text-left">
          <h1 className="text-3xl font-semibold text-slate-900">Text Reuse Lab (Traditional)</h1>
          <p className="text-slate-600 text-base sm:max-w-3xl">
            Run comparative text reuse algorithms (n-gram, GST, TF-IDF) between notebook pairs. Configure parameters, view similarity scores, and explore detected reuse instances.
          </p>
        </header>

        <div className="rounded-3xl border border-slate-200 bg-white/90 shadow-sm p-6 space-y-4">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-2 uppercase tracking-wide">Algorithm</label>
              <select 
                value={alg} 
                onChange={(e)=>setAlg(e.target.value)} 
                className="w-full rounded-full border border-slate-200 bg-white px-4 py-2.5 text-sm text-slate-700 shadow-sm transition focus:border-indigo-300 focus:outline-none focus:ring-2 focus:ring-indigo-100"
              >
                <option value="ngram">N-gram (Token overlap)</option>
                <option value="gst">GST (Greedy String Tiling)</option>
                <option value="tfidf">TF-IDF (Cosine similarity)</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-2 uppercase tracking-wide">Configuration</label>
              <select 
                value={configId} 
                onChange={(e)=>setConfigId(e.target.value)} 
                className="w-full rounded-full border border-slate-200 bg-white px-4 py-2.5 text-sm text-slate-700 shadow-sm transition focus:border-indigo-300 focus:outline-none focus:ring-2 focus:ring-indigo-100"
              >
                {configs.map(c => <option key={c.config_id} value={c.config_id}>{c.config_id} — {c.description}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-2 uppercase tracking-wide">Notebook Pair</label>
              <div className="flex gap-2">
                <select 
                  value={nb1} 
                  onChange={(e)=>setNb1(e.target.value)} 
                  className="flex-1 rounded-full border border-slate-200 bg-white px-3 py-2.5 text-sm text-slate-700 shadow-sm transition focus:border-indigo-300 focus:outline-none focus:ring-2 focus:ring-indigo-100"
                >
                  <option value="">NB1…</option>
                  {notebooks.map(n => <option key={n} value={n}>{n}</option>)}
                </select>
                <select 
                  value={nb2} 
                  onChange={(e)=>setNb2(e.target.value)} 
                  className="flex-1 rounded-full border border-slate-200 bg-white px-3 py-2.5 text-sm text-slate-700 shadow-sm transition focus:border-indigo-300 focus:outline-none focus:ring-2 focus:ring-indigo-100"
                >
                  <option value="">NB2…</option>
                  {notebooks.map(n => <option key={n} value={n}>{n}</option>)}
                </select>
              </div>
            </div>
          </div>

          <button 
            onClick={fetchOrRun} 
            disabled={loading || !nb1 || !nb2 || !configId} 
            className="w-full sm:w-auto inline-flex items-center justify-center gap-2 rounded-full bg-indigo-600 px-6 py-3 text-sm font-medium text-white shadow-sm transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:bg-indigo-300"
          >
            {loading ? (cacheStatus === 'running' ? 'Running analysis…' : 'Loading…') : 'Run / Load Results'}
          </button>
        </div>

        {loading && !results && (
          <div className="space-y-4">
            <div className="rounded-3xl border border-slate-200 bg-white/90 p-6 shadow-sm animate-pulse">
              <div className="h-4 bg-slate-200 rounded w-1/3 mb-4"></div>
              <div className="h-3 bg-slate-100 rounded w-2/3"></div>
            </div>
            <div className="grid gap-6 md:grid-cols-2">
              <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm animate-pulse">
                <div className="h-3 bg-slate-200 rounded w-1/2 mb-4"></div>
                <div className="h-32 bg-slate-100 rounded-2xl"></div>
              </div>
              <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm animate-pulse">
                <div className="h-3 bg-slate-200 rounded w-1/2 mb-4"></div>
                <div className="h-32 bg-slate-100 rounded-2xl"></div>
              </div>
            </div>
            <div className="text-center text-sm text-slate-500">
              {cacheStatus === 'run' ? 'Running text reuse analysis...' : 'Loading results...'}
            </div>
          </div>
        )}

        {error && (
          <div className="rounded-2xl border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700">
            <strong className="font-semibold">Error: </strong>{error}
          </div>
        )}

      {results && (
        <div className="space-y-6">
          <div className="flex items-center justify-between rounded-2xl border border-slate-200 bg-white/90 p-4 shadow-sm">
            <div className="flex items-center gap-3">
              <span className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-medium ${badgeClasses[alg]}`}>
                {alg.toUpperCase()}
              </span>
              <span className="text-sm text-slate-600">
                {instances.length} {instances.length === 1 ? 'instance' : 'instances'} detected
              </span>
              {cacheStatus === 'loaded' && instances.length > 0 && (
                <span className="text-xs text-emerald-600 font-medium">(Loaded from cache)</span>
              )}
            </div>
            {score !== null && (
              <div className="text-sm text-slate-700">
                <span className="text-slate-500">Similarity: </span>
                <span className="font-semibold">{fmt(score, 3)}</span>
              </div>
            )}
          </div>

          {instances.length > 0 && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-2">
                  <div className="text-sm text-slate-600">
                    Instance <span className="font-semibold">{idx+1}</span> of <span className="font-semibold">{instances.length}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 w-48 bg-slate-200 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-indigo-500 transition-all duration-300 rounded-full"
                        style={{ width: `${((idx + 1) / instances.length) * 100}%` }}
                      />
                    </div>
                    <span className="text-xs text-slate-400">Use ← → keys</span>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <button 
                    disabled={idx<=0} 
                    onClick={()=>setIdx(i=>Math.max(0,i-1))} 
                    className="rounded-full border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    ← Previous
                  </button>
                  <button 
                    disabled={idx>=instances.length-1} 
                    onClick={()=>setIdx(i=>Math.min(instances.length-1,i+1))} 
                    className="rounded-full border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    Next →
                  </button>
                </div>
              </div>

              <div className="grid gap-6 md:grid-cols-2">
                <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-medium text-slate-500 uppercase tracking-wide">Notebook {current?.notebook1}</span>
                    <span className="text-xs text-slate-500">Page {current?.segment1_page_key}</span>
                  </div>
                  <div className="text-sm font-semibold text-slate-900 mb-3">Segment 1</div>
                  <div className="text-sm leading-relaxed whitespace-pre-wrap bg-slate-50 p-4 rounded-2xl text-slate-800 max-h-96 overflow-auto">
                    {current?.segment1_text}
                  </div>
                  {renderMetadata(current?.segment1_metadata, '📊 View metadata')}
                </div>
                <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-medium text-slate-500 uppercase tracking-wide">Notebook {current?.notebook2}</span>
                    <span className="text-xs text-slate-500">Page {current?.segment2_page_key}</span>
                  </div>
                  <div className="text-sm font-semibold text-slate-900 mb-3">Segment 2</div>
                  <div className="text-sm leading-relaxed whitespace-pre-wrap bg-slate-50 p-4 rounded-2xl text-slate-800 max-h-96 overflow-auto">
                    {current?.segment2_text}
                  </div>
                  {renderMetadata(current?.segment2_metadata, '📊 View metadata')}
                </div>
              </div>

              {results?.summary_metrics && Object.keys(results.summary_metrics).length > 0 && (
                <div className="rounded-3xl border border-indigo-100 bg-indigo-50/50 p-5 shadow-sm">
                  <div className="text-sm font-semibold text-indigo-900 mb-3">Summary Metrics</div>
                  <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 text-xs">
                    {Object.entries(results.summary_metrics).slice(0, 12).map(([key, value]) => (
                      <div key={key} className="flex flex-col gap-1 rounded-xl bg-white/60 p-3">
                        <span className="text-indigo-600 font-medium">{key.replace(/_/g, ' ')}</span>
                        <span className="text-slate-900 font-mono font-semibold">{fmt(value)}</span>
                      </div>
                    ))}
                  </div>
                  {Object.keys(results.summary_metrics).length > 12 && (
                    <details className="mt-3">
                      <summary className="cursor-pointer text-xs text-indigo-600 hover:text-indigo-700 font-medium">View all metrics ({Object.keys(results.summary_metrics).length})</summary>
                      <div className="mt-3 rounded-2xl bg-white/60 p-4 text-xs space-y-2 max-h-64 overflow-auto">
                        {Object.entries(results.summary_metrics).slice(12).map(([key, value]) => (
                          <div key={key} className="flex justify-between gap-3">
                            <span className="text-slate-600 font-medium">{key.replace(/_/g, ' ')}</span>
                            <span className="text-slate-900 font-mono">{fmt(value)}</span>
                          </div>
                        ))}
                      </div>
                    </details>
                  )}
                </div>
              )}
            </div>
          )}

          {instances.length === 0 && (
            <div className="rounded-3xl border border-amber-200 bg-amber-50 p-8 text-center shadow-sm">
              <p className="text-amber-900 font-medium">No text reuse instances found for this configuration.</p>
              <p className="text-sm text-amber-700 mt-2">Try adjusting the algorithm, configuration, or notebook pair.</p>
            </div>
          )}
        </div>
      )}
      </div>
    </div>
  )
}


