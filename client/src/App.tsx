import { useEffect } from 'react';
import { Routes, Route } from 'react-router-dom';
import { AppShell } from '@mantine/core';
import { useGame } from './GameContext';
import { useSSE } from './hooks/useSSE';
import { useStatus } from './hooks/useStatus';
import { useAnalysis } from './hooks/useAnalysis';
import TabBar from './components/Layout/TabBar';
import GamesView from './views/GamesView';
import AnalysisView from './views/AnalysisView';
import TrainingView from './views/TrainingView';
import LogsView from './views/LogsView';
import FullscreenOverlay from './components/Fullscreen/FullscreenOverlay';
import Docs from './pages/Docs';

/** Main training dashboard */
function Dashboard() {
  const { activeTab, setActiveTab } = useGame();

  const renderView = () => {
    switch (activeTab) {
      case 'games':
        return <GamesView />;
      case 'analysis':
        return <AnalysisView />;
      case 'training':
        return <TrainingView />;
      case 'logs':
        return <LogsView />;
      default:
        return <GamesView />;
    }
  };

  return (
    <AppShell
      header={{ height: 50 }}
      styles={{
        main: {
          background: '#0D1117',
          paddingTop: '50px',
          paddingLeft: 0,
          paddingRight: 0,
        },
      }}
    >
      <AppShell.Header>
        <TabBar activeTab={activeTab} onTabChange={setActiveTab} />
      </AppShell.Header>

      <AppShell.Main>
        <div style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 50px)' }}>
          {renderView()}
        </div>
      </AppShell.Main>
    </AppShell>
  );
}

export default function App() {
  const { state } = useGame();

  // Initialize hooks (keep alive across all routes)
  useSSE();
  useStatus();
  useAnalysis();

  useEffect(() => {
    if (state.autoFollow && state.allMoves.length > 0) {
      // The view index is already updated in the SSE hook
    }
  }, [state.allMoves.length, state.autoFollow]);

  return (
    <>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/docs" element={<Docs />} />
      </Routes>
      <FullscreenOverlay />
    </>
  );
}
