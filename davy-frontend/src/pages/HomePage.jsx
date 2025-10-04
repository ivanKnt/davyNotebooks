import { Link } from 'react-router-dom'

const featureCards = [
  {
    title: 'Preprocessing Pipeline',
    description: 'Extract and normalize text from TEI XML sources. Generate page-to-text mappings, entity annotations, and classification metadata for downstream analysis.',
    to: '/preprocessing',
    status: 'Available',
    tone: 'green'
  },
  {
    title: 'Classification Reader',
    description: 'Browse notebooks with interactive per-page classifications. View consensus labels, percentage breakdowns, and explore thematic patterns across Davy\'s writing.',
    to: '/classification',
    status: 'Available',
    tone: 'green'
  },
  {
    title: 'Poetry Explorer (Traditional)',
    description: 'Discover pages classified as poetry across the entire corpus. Filter by notebook, view classification confidence, and read poetic excerpts in context.',
    to: '/poetry/traditional',
    status: 'Available',
    tone: 'green'
  },
  {
    title: 'Poetry Explorer (Advanced)',
    description: 'Machine learning-based poetry detection using transformer models. Enhanced accuracy and explainability features under development.',
    status: 'Coming Soon',
    disabled: true,
    tone: 'yellow'
  },
  {
    title: 'Text Reuse Lab (Traditional)',
    description: 'Run n-gram, GST, or TF-IDF algorithms to detect text reuse between notebook pairs. Configure parameters, view similarity metrics, and export results.',
    to: '/text-reuse/traditional',
    status: 'Available',
    tone: 'green'
  },
  {
    title: 'Text Reuse Lab (Advanced)',
    description: 'Semantic similarity detection with BERT embeddings, cross-notebook network visualization, and advanced filtering. Planned for future release.',
    status: 'Coming Soon',
    disabled: true,
    tone: 'yellow'
  }
]

export default function HomePage() {
  return (
    <div className="bg-gradient-to-b from-slate-100 via-white to-slate-100">
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,_rgba(79,_70,_229,_0.12)_0,_transparent_55%)]" />
        <div className="max-w-6xl mx-auto px-4 pt-16 pb-20 relative">
          <div className="grid gap-10 lg:grid-cols-[minmax(0,_1fr)_320px] items-center">
            <div className="space-y-6">
              <span className="inline-flex items-center gap-2 rounded-full bg-indigo-50 px-3 py-1 text-sm font-medium text-indigo-600 ring-1 ring-indigo-100">
                Humphry Davy Digital Humanities Project
              </span>
              <h1 className="text-4xl sm:text-5xl font-semibold tracking-tight text-slate-900">
                Mapping the intellectual currents of the <span className="text-indigo-600">Davy Notebooks</span> project.
              </h1>
              <p className="text-lg text-slate-600 leading-relaxed">
                A unified laboratory for preprocessing, classification, poetry analysis, and text reuse experiments. Dive into annotated notebooks, run comparative algorithms, and trace literary influence across centuries.
              </p>
              <div className="flex flex-wrap gap-3">
                <Link to="/text-reuse/traditional" className="inline-flex items-center gap-2 rounded-full bg-indigo-600 px-5 py-2.5 text-sm font-medium text-white shadow-lg shadow-indigo-600/20 transition hover:bg-indigo-500">
                  Explore Text Reuse Workbench
                </Link>
                <Link to="/classification" className="inline-flex items-center gap-2 rounded-full bg-white px-5 py-2.5 text-sm font-medium text-slate-700 ring-1 ring-slate-200 transition hover:ring-indigo-200 hover:text-indigo-600">
                  Open Classification Reader
                </Link>
              </div>
            </div>
            <div className="relative hidden lg:block">
              <div className="aspect-[4/5] overflow-hidden rounded-3xl shadow-2xl shadow-indigo-900/20 ring-1 ring-white/60">
                <img
                  src="/images/sir_humphry_davy.jpg"
                  alt="Portrait of Sir Humphry Davy seated at a writing desk"
                  className="h-full w-full object-cover"
                />
              </div>
              <div className="mt-4 flex items-start justify-between text-xs text-slate-500">
                <span>Portrait of Sir Humphry Davy, c.1800s</span>
                <span>Public domain</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="max-w-6xl mx-auto px-4 pb-20">
        <div className="mb-12 text-center">
          <h2 className="text-2xl font-semibold text-slate-900">Research Workbench</h2>
          <p className="mt-3 text-slate-600 max-w-3xl mx-auto">
            Each workspace integrates directly with Python scripts in the repository, offering reproducible pipelines and curated outputs for scholars and developers alike.
          </p>
        </div>
        <div className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3">
          {featureCards.map((feature) => {
            const tone = feature.tone || 'slate'
            const toneMap = {
              green: {
                ring: 'ring-emerald-100',
                badge: 'bg-emerald-100 text-emerald-700',
                hover: 'hover:ring-emerald-200'
              },
              yellow: {
                ring: 'ring-amber-100',
                badge: 'bg-amber-100 text-amber-700',
                hover: 'hover:ring-amber-200'
              },
              slate: {
                ring: 'ring-slate-100',
                badge: 'bg-slate-100 text-slate-700',
                hover: 'hover:ring-slate-200'
              }
            }[tone]

            const CardWrapper = feature.disabled ? 'div' : Link
            const cardClasses = `group relative flex h-full flex-col rounded-2xl border border-slate-200 bg-white/90 p-6 shadow-sm transition ${
              feature.disabled
                ? 'opacity-70'
                : 'hover:shadow-lg hover:-translate-y-1 hover:border-indigo-200'
            }`

            return (
              <CardWrapper
                key={feature.title}
                to={feature.to || '#'}
                className={cardClasses}
              >
                <div className={`absolute inset-0 rounded-2xl ring-1 opacity-0 group-hover:opacity-100 transition ${toneMap.ring}`} />
                <div className="relative z-10 flex-1 space-y-4">
                  <div className="flex justify-between items-start gap-4">
                    <h3 className="text-lg font-semibold text-slate-900">{feature.title}</h3>
                    <span className={`rounded-full px-3 py-1 text-xs font-medium ${toneMap.badge}`}>
                      {feature.status}
                    </span>
                  </div>
                  <p className="text-sm leading-relaxed text-slate-600">
                    {feature.description}
                  </p>
                </div>
                {feature.disabled && (
                  <div className="relative z-10 mt-4 text-xs font-medium uppercase tracking-wide text-slate-400">
                    Development in progress
                  </div>
                )}
              </CardWrapper>
            )
          })}
        </div>
      </section>
    </div>
  )
}


