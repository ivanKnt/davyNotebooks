import { useEffect, useState, useMemo } from 'react'
import { runClassification, listClassificationNotebooks, getNotebookClassification, getPageClassification } from '../services/api.js'
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip } from 'recharts'

export default function ClassificationPage() {
  const [notebooks, setNotebooks] = useState([])
  const [selected, setSelected] = useState('')
  const [bookData, setBookData] = useState(null)
  const [page, setPage] = useState('')
  const [pageData, setPageData] = useState(null)
  const [running, setRunning] = useState(false)
  const [runError, setRunError] = useState('')

  const load = async () => {
    const res = await listClassificationNotebooks()
    setNotebooks(res.data)
  }
  useEffect(() => { load() }, [])

  const onRun = async () => {
    setRunning(true)
    setRunError('')
    try {
      const res = await runClassification()
      await load()
      if (res?.data?.output) {
        console.debug('classification output', res.data.output)
      }
    } catch (e) {
      setRunError(e?.response?.data?.details || e?.response?.data?.message || String(e))
    } finally { setRunning(false) }
  }
  const loadNotebook = async (id) => {
    setSelected(id); setBookData(null); setPageData(null); setPage('')
    if (!id) return
    const res = await getNotebookClassification(id)
    setBookData(res.data)
  }
  const loadPage = async () => {
    if (!selected || !page) return
    const res = await getPageClassification(selected, page)
    setPageData(res.data)
  }

  // Derive a distribution and consensus from pageData.classification
  const { dist, consensusKey } = useMemo(() => {
    const cls = pageData?.classification
    if (!cls || typeof cls !== 'object') return { dist: [], consensusKey: '' }
    // If classification is already a mapping of label -> number (0..1), use it
    const entries = Object.entries(cls).filter(([, v]) => typeof v === 'number' && isFinite(v))
    if (entries.length > 0) {
      const sorted = entries.sort((a, b) => b[1] - a[1])
      return {
        dist: sorted.map(([name, value]) => ({ name, value: Number(value) })),
        consensusKey: sorted[0]?.[0] || ''
      }
    }
    // Else if it has a 'consensus' string and a 'percentages' map
    const consensus = typeof cls.consensus === 'string' ? cls.consensus : ''
    const pct = cls.percentages && typeof cls.percentages === 'object' ? Object.entries(cls.percentages) : []
    const sorted2 = pct.filter(([, v]) => typeof v === 'number').sort((a, b) => b[1] - a[1])
    return {
      dist: sorted2.map(([name, value]) => ({ name, value: Number(value) })),
      consensusKey: consensus || (sorted2[0]?.[0] || '')
    }
  }, [pageData])

  const fmtPct = (v) => `${(v * 100).toFixed(1)}%`

  return (
    <div className="bg-gradient-to-b from-white to-slate-100 min-h-screen">
      <div className="max-w-6xl mx-auto px-4 py-12 space-y-8">
        <header className="space-y-3 text-center sm:text-left">
          <h1 className="text-3xl font-semibold text-slate-900">Classification Reader</h1>
          <p className="text-slate-600 text-base sm:max-w-3xl">
            Browse thematic classifications for each notebook and page. Explore consensus labels, percentage distributions, and the full text content.
          </p>
        </header>

        <div className="rounded-3xl border border-slate-200 bg-white/90 shadow-sm p-6">
          <div className="flex flex-wrap items-center gap-3 mb-4">
            <button 
              onClick={onRun} 
              disabled={running} 
              className="inline-flex items-center gap-2 rounded-full bg-indigo-600 px-5 py-2.5 text-sm font-medium text-white shadow-sm transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:bg-indigo-300"
            >
              {running ? 'Processing Classifications…' : 'Run Classification Processing'}
            </button>
            <select 
              value={selected} 
              onChange={(e) => loadNotebook(e.target.value)} 
              className="flex-1 min-w-[280px] rounded-full border border-slate-200 bg-white px-4 py-2.5 text-sm text-slate-700 shadow-sm transition focus:border-indigo-300 focus:outline-none focus:ring-2 focus:ring-indigo-100"
            >
              <option value="">Select a notebook to explore…</option>
              {notebooks.map(n => <option key={n.id} value={n.id}>{n.id} — {n.title} ({n.consensus})</option>)}
            </select>
          </div>
          {runError && (
            <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-2xl p-3">{runError}</div>
          )}
        </div>

      {bookData && (
        <div className="grid gap-6 lg:grid-cols-2">
          <div className="rounded-3xl border border-emerald-100 bg-emerald-50/50 p-6 shadow-sm">
            <h3 className="text-lg font-semibold text-emerald-900 mb-4">Notebook Overview</h3>
            <div className="space-y-3 text-sm text-emerald-800">
              <div className="flex justify-between">
                <span className="text-emerald-600 font-medium">Notebook:</span>
                <span className="font-semibold">{bookData.notebook_title || selected}</span>
              </div>
              {bookData.consensus_book && (
                <div className="flex justify-between">
                  <span className="text-emerald-600 font-medium">Overall Consensus:</span>
                  <span className="inline-flex items-center rounded-full bg-emerald-600 px-3 py-1 text-xs font-medium text-white">{bookData.consensus_book}</span>
                </div>
              )}
            </div>
            <details className="mt-4">
              <summary className="cursor-pointer text-sm text-emerald-600 hover:text-emerald-700 font-medium">View raw classification data</summary>
              <pre className="mt-2 text-xs bg-white/60 p-3 rounded-2xl overflow-auto max-h-48 text-emerald-900">{JSON.stringify(bookData, null, 2)}</pre>
            </details>
          </div>
          <div className="rounded-3xl border border-indigo-100 bg-indigo-50/50 p-6 shadow-sm">
            <h3 className="text-lg font-semibold text-indigo-900 mb-4">Page Lookup</h3>
            <div className="flex gap-2 mb-4">
              <input 
                value={page} 
                onChange={(e)=>setPage(e.target.value)} 
                placeholder="Page number" 
                className="flex-1 rounded-full border border-indigo-200 bg-white px-4 py-2 text-sm text-slate-700 shadow-sm transition focus:border-indigo-300 focus:outline-none focus:ring-2 focus:ring-indigo-100" 
              />
              <button 
                onClick={loadPage} 
                className="rounded-full bg-indigo-600 px-5 py-2 text-sm font-medium text-white shadow-sm transition hover:bg-indigo-500"
              >
                Load Page
              </button>
            </div>
            {pageData && (
              <div className="space-y-4">
                <div className="flex items-center justify-between text-sm text-indigo-800">
                  <div>
                    <span className="text-indigo-600 font-medium">Notebook:</span> {pageData.notebook_id} — <span className="text-indigo-600 font-medium">Page:</span> {pageData.page_number}
                  </div>
                  {dist.length > 0 && (
                    <span className="inline-flex items-center rounded-full bg-indigo-600 px-3 py-1 text-xs font-medium text-white">
                      {consensusKey} ({fmtPct(dist.find(d => d.name === consensusKey)?.value || 0)})
                    </span>
                  )}
                </div>

                {dist.length > 0 && (
                  <div className="space-y-4">
                    <div className="rounded-2xl border border-slate-200 bg-white p-4">
                      <div className="text-sm font-medium text-slate-700 mb-3">Classification Distribution</div>
                      <ResponsiveContainer width="100%" height={200}>
                        <BarChart data={dist} margin={{ top: 8, right: 8, left: 0, bottom: 24 }}>
                          <XAxis dataKey="name" angle={-30} textAnchor="end" height={40} interval={0} tick={{ fontSize: 11 }} />
                          <YAxis tickFormatter={(v)=>`${(v*100).toFixed(0)}%`} width={36} />
                          <Tooltip formatter={(v)=>fmtPct(v)} />
                          <Bar dataKey="value" fill="#4f46e5" radius={[4,4,0,0]} />
                        </BarChart>
                      </ResponsiveContainer>
                      <div className="mt-3 space-y-1.5 text-sm">
                        {dist.map(d => (
                          <div key={d.name} className="flex justify-between items-center">
                            <span className={d.name === consensusKey ? 'font-semibold text-indigo-900' : 'text-slate-700'}>{d.name}</span>
                            <span className={d.name === consensusKey ? 'font-semibold text-indigo-600' : 'text-slate-500'}>{fmtPct(d.value)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                    <div className="rounded-2xl border border-slate-200 bg-white p-4">
                      <div className="text-sm font-medium text-slate-700 mb-2">Page Content</div>
                      <div className="text-sm leading-relaxed whitespace-pre-wrap max-h-96 overflow-auto bg-slate-50 p-4 rounded-xl text-slate-800">{pageData.text}</div>
                    </div>
                  </div>
                )}

                {dist.length === 0 && (
                  <pre className="text-xs bg-white/60 p-3 rounded-2xl overflow-auto max-h-48 text-slate-900">{JSON.stringify(pageData, null, 2)}</pre>
                )}
              </div>
            )}
          </div>
        </div>
      )}
      </div>
    </div>
  )
}


