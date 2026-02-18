import { Routes, Route } from 'react-router-dom';
import Layout from './components/layout/Layout';
import OverviewPage from './pages/OverviewPage';
import ExplorerPage from './pages/ExplorerPage';
import SimilarityPage from './pages/SimilarityPage';
import ForecastingPage from './pages/ForecastingPage';
import ComparisonPage from './pages/ComparisonPage';

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<OverviewPage />} />
        <Route path="/explorer" element={<ExplorerPage />} />
        <Route path="/similarity" element={<SimilarityPage />} />
        <Route path="/forecasting" element={<ForecastingPage />} />
        <Route path="/comparison" element={<ComparisonPage />} />
      </Route>
    </Routes>
  );
}
