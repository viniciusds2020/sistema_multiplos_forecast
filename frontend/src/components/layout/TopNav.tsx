import { NavLink } from 'react-router-dom';

const navItems = [
  { to: '/', label: 'Visão Geral' },
  { to: '/explorer', label: 'Exploração de Dados' },
  { to: '/similarity', label: 'Similaridade' },
  { to: '/forecasting', label: 'Previsão' },
  { to: '/comparison', label: 'Comparação' },
];

export default function TopNav() {
  return (
    <header className="sticky top-0 z-40">
      <div className="bg-brand-600 text-white text-xs">
        <div className="max-w-[1400px] mx-auto px-6 py-2 flex items-center justify-end">
          <div className="uppercase tracking-[.2em] text-[10px]">
            Veiling Online
          </div>
        </div>
      </div>

      <div className="bg-white/95 backdrop-blur border-b border-brand-100 shadow-sm">
        <div className="max-w-[1400px] mx-auto px-6 py-4 flex flex-wrap items-center justify-between gap-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-brand-500/10 border border-brand-200 flex items-center justify-center text-brand-700 font-bold">
              FF
            </div>
            <div>
              <div className="text-xl font-semibold text-brand-800 tracking-tight">ForecastForge</div>
              <div className="text-[11px] uppercase tracking-[.3em] text-brand-400">Forecast Dashboard</div>
            </div>
          </div>

          <nav className="flex flex-wrap items-center gap-6 text-xs font-semibold text-gray-600">
            {navItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === '/'}
                className={({ isActive }) =>
                  `nav-link pb-1 ${isActive ? 'active' : ''}`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>

          <div className="flex items-center gap-3">
            <button className="text-xs uppercase tracking-[.2em] px-4 py-2 border border-brand-200 rounded-full text-brand-700 hover:text-brand-800 hover:border-brand-400 transition">
              Relatórios
            </button>
            <button className="text-xs uppercase tracking-[.2em] px-4 py-2 rounded-full bg-brand-500 text-white shadow hover:bg-brand-600 transition">
              Online
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}
