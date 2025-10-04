import { useState } from 'react'
import { runPreprocessing, getPreprocessingStatus } from '../services/api.js'

export default function PreprocessingPage() {
  const [status, setStatus] = useState(null)
  const [running, setRunning] = useState(false)
  const [output, setOutput] = useState('')

  const check = async () => {
    const res = await getPreprocessingStatus()
    setStatus(res.data)
  }
  const run = async () => {
    setRunning(true)
    setOutput('')
    try {
      const res = await runPreprocessing()
      setOutput(res.data.output || '')
      await check()
    } catch (e) {
      setOutput(e?.response?.data?.details || String(e))
    } finally {
      setRunning(false)
    }
  }
  const prerequisites = [
    {
      label: 'TEI XML sources located under `items/`',
      status: 'Ready'
    },
    {
      label: 'Preprocessing outputs saved to `preprocessing/`',
      status: status?.status === 'completed' ? 'Completed' : 'Pending'
    },
    {
      label: 'Classification, poetry, and text reuse depend on these files',
      status: 'Dependency'
    }
  ]

  return (
    <div className="bg-gradient-to-b from-white to-slate-100">
      <div className="max-w-5xl mx-auto px-4 py-12 space-y-12">
        <header className="space-y-3 text-center sm:text-left">
          <h1 className="text-3xl font-semibold text-slate-900">Preprocessing Pipeline</h1>
          <p className="text-slate-600 text-base sm:max-w-2xl">
            Extract plain text, entities, and per-page metadata from TEI notebook scans. This is the foundation for every downstream interface in the Davy Notebooks platform.
          </p>
        </header>

        <section className="grid gap-6 lg:grid-cols-[minmax(0,_3fr)_minmax(0,_2fr)]">
          <div className="rounded-3xl border border-slate-200 bg-white/90 p-6 shadow-sm">
            <div className="flex flex-wrap gap-3 items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold text-slate-900">Run Pipeline</h2>
                <p className="text-sm text-slate-500">Execute `preprocess_files.py` via the Flask API runner.</p>
              </div>
              <span className={`inline-flex items-center gap-2 rounded-full px-4 py-1 text-sm font-medium ${running ? 'bg-amber-100 text-amber-700' : status?.status === 'completed' ? 'bg-emerald-100 text-emerald-700' : 'bg-slate-100 text-slate-600'}`}>
                {running ? 'Running…' : status?.status === 'completed' ? 'Completed' : 'Idle'}
              </span>
            </div>

            <div className="mt-5 flex flex-wrap gap-3">
              <button
                onClick={run}
                disabled={running}
                className="inline-flex items-center gap-2 rounded-full bg-indigo-600 px-5 py-2.5 text-sm font-medium text-white shadow-sm transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:bg-indigo-300"
              >
                {running ? 'Processing…' : 'Run Preprocessing'}
              </button>
              <button
                onClick={check}
                className="inline-flex items-center gap-2 rounded-full bg-white px-5 py-2.5 text-sm font-medium text-slate-700 ring-1 ring-slate-200 transition hover:text-indigo-600 hover:ring-indigo-200"
              >
                Refresh Status
              </button>
            </div>

            {status && (
              <div className="mt-6 rounded-2xl border border-slate-200 bg-slate-50 p-4 text-sm text-slate-700">
                <div className="flex items-center justify-between">
                  <span className="font-medium">Latest status</span>
                  <span className="text-xs text-slate-500">{new Date().toLocaleString()}</span>
                </div>
                <pre className="mt-3 max-h-48 overflow-auto whitespace-pre-wrap text-xs text-slate-600">
                  {JSON.stringify(status, null, 2)}
                </pre>
              </div>
            )}

            {output && (
              <details className="mt-6 rounded-2xl border border-slate-200 bg-white p-4 text-sm text-slate-700">
                <summary className="cursor-pointer font-medium">Execution log</summary>
                <pre className="mt-3 max-h-60 overflow-auto whitespace-pre-wrap text-xs text-slate-600">
                  {output}
                </pre>
              </details>
            )}
          </div>

          <aside className="space-y-4">
            <div className="rounded-3xl border border-indigo-100 bg-indigo-50 p-6 text-sm text-indigo-900">
              <h3 className="text-base font-semibold">What happens when you run this?</h3>
              <ul className="mt-4 space-y-2 text-indigo-800">
                <li>• Parses TEI XML files and generates `page_to_text.json`.</li>
                <li>• Extracts named entities per page (`page_to_entities.json`).</li>
                <li>• Normalises classification metadata for downstream readers.</li>
              </ul>
            </div>

            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <h3 className="text-base font-semibold text-slate-900">Pipeline checklist</h3>
              <ul className="mt-4 space-y-3 text-sm text-slate-600">
                {prerequisites.map((item) => (
                  <li key={item.label} className="flex items-center justify-between gap-3">
                    <span>{item.label}</span>
                    <span className="text-xs font-medium uppercase tracking-wide text-slate-500">{item.status}</span>
                  </li>
                ))}
              </ul>
            </div>
          </aside>
        </section>
      </div>
    </div>
  )
}


