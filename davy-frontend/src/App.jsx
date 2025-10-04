import { Routes, Route, Link, NavLink } from 'react-router-dom'
import HomePage from './pages/HomePage.jsx'
import PreprocessingPage from './pages/PreprocessingPage.jsx'
import ClassificationPage from './pages/ClassificationPage.jsx'
import PoetryTraditionalPage from './pages/PoetryTraditionalPage.jsx'
import PoetryAdvancedPage from './pages/PoetryAdvancedPage.jsx'
import TextReuseTraditionalPage from './pages/TextReuseTraditionalPage.jsx'
import TextReuseAdvancedPage from './pages/TextReuseAdvancedPage.jsx'
import InventoryPage from './pages/InventoryPage.jsx'

function NavItem({ to, children, disabled, description }) {
  if (disabled) {
    return (
      <span className="px-3 py-2 text-sm text-slate-400 cursor-not-allowed flex flex-col items-start">
        <span>{children}</span>
        {description && <span className="text-[10px] uppercase tracking-wide text-slate-500">{description}</span>}
      </span>
    )
  }
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `px-3 py-2 text-sm transition-colors duration-200 rounded-md flex flex-col items-start ${
          isActive ? 'bg-white/80 text-slate-900 shadow-sm' : 'text-slate-200 hover:text-white hover:bg-white/10'
        }`
      }
    >
      <span>{children}</span>
      {description && <span className="text-[10px] uppercase tracking-wide text-slate-300">{description}</span>}
    </NavLink>
  )
}

export default function App() {
  return (
    <div className="min-h-screen flex flex-col bg-slate-900 text-slate-100">
      <header className="border-b border-slate-800 bg-gradient-to-r from-slate-900 via-slate-900 to-indigo-900/60">
        <div className="max-w-6xl mx-auto px-4 py-4 flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <Link to="/" className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-full bg-indigo-500/40 ring-1 ring-indigo-300/40 flex items-center justify-center font-semibold text-indigo-100">
              DN
            </div>
            <div>
              <p className="text-xl font-semibold tracking-tight">Davy Notebooks Project</p>
              <p className="text-xs text-slate-300 uppercase tracking-[0.2em]">Lancaster University</p>
            </div>
          </Link>
          <nav className="flex flex-wrap items-center gap-3">
            <NavItem to="/preprocessing" description="Pipeline">Preprocessing</NavItem>
            <NavItem to="/classification" description="Reader">Classification</NavItem>
            <NavItem to="/poetry/traditional" description="Traditional">Poetry</NavItem>
            <NavItem to="/poetry/advanced" disabled description="Coming Soon">Poetry (Advanced)</NavItem>
            <NavItem to="/text-reuse/traditional" description="Experiments">Text Reuse</NavItem>
            <NavItem to="/text-reuse/advanced" disabled description="Coming Soon">Text Reuse (Advanced)</NavItem>
            <NavItem to="/inventory" disabled description="Planned">Inventory</NavItem>
          </nav>
        </div>
      </header>

      <main className="flex-1 bg-slate-50 text-slate-900">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/preprocessing" element={<PreprocessingPage />} />
          <Route path="/classification" element={<ClassificationPage />} />
          <Route path="/poetry/traditional" element={<PoetryTraditionalPage />} />
          <Route path="/poetry/advanced" element={<PoetryAdvancedPage />} />
          <Route path="/text-reuse/traditional" element={<TextReuseTraditionalPage />} />
          <Route path="/text-reuse/advanced" element={<TextReuseAdvancedPage />} />
          <Route path="/inventory" element={<InventoryPage />} />
        </Routes>
      </main>

      <footer className="border-t border-slate-800 bg-slate-900 text-center text-xs text-slate-400 py-6">
        © {new Date().getFullYear()} Davy Notebooks Project · Lancaster University
      </footer>
    </div>
  )
}


