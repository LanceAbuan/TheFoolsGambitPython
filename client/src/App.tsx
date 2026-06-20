import { useEffect, useState } from 'react';
import { Routes, Route } from 'react-router-dom';
import { AppShell, Drawer } from '@mantine/core';
import { useGame } from './GameContext';
import { useSSE } from './hooks/useSSE';
import { useStatus } from './hooks/useStatus';
import { useAnalysis } from './hooks/useAnalysis';
import TopBar from './components/Layout/TopBar';
import PlayerInfoBar from './components/Board/PlayerInfoBar';
import LiveBoard from './components/Board/LiveBoard';
import BoardNav from './components/Board/BoardNav';
import EvalBar from './components/Board/EvalBar';
import MoveList from './components/Analysis/MoveList';
import AnalysisTable from './components/Analysis/AnalysisTable';
import MetricsGrid from './components/Metrics/MetricsGrid';
import SSEEventLog from './components/SSE/SSEEventLog';
import RecentGames from './components/Games/RecentGames';
import SideBoard from './components/Games/SideBoard';
import FullscreenOverlay from './components/Fullscreen/FullscreenOverlay';
import Docs from './pages/Docs';

/** Main training dashboard */
function Dashboard() {
  const { state } = useGame();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const whiteToMove = state.allMoves.length % 2 === 0;

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
        <TopBar onToggleSidebar={() => setSidebarOpen(!sidebarOpen)} />
      </AppShell.Header>

      <AppShell.Main>
        <div className="main-layout" style={{ padding: 16 }}>
          {/* Center column */}
          <div className="main-column">
            <PlayerInfoBar
              name="Self-Play (NN)"
              detail="vs Self"
              isTurn={!whiteToMove}
              color="top"
            />
            <div className="board-row">
              <EvalBar />
              <LiveBoard />
            </div>
            <PlayerInfoBar
              name="Fool's Gambit AI"
              detail="Training"
              isTurn={whiteToMove}
              color="bottom"
            />
            <BoardNav />
          </div>

          {/* Side games column — 3×3 grid */}
          <div className="side-games-column">
            <div className="side-boards-grid">
              {[1, 2, 3, 4, 5, 6, 7, 8, 9].map((id) => (
                <SideBoard key={id} gameId={id} />
              ))}
            </div>
          </div>
        </div>
      </AppShell.Main>

      {/* Sidebar as collapsible Drawer */}
      <Drawer
        opened={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        position="right"
        size={360}
        title="Training Stats"
        styles={{
          title: { color: '#C9D1D9', fontWeight: 700, fontSize: '14px' },
          header: { background: '#161B22', borderBottom: '1px solid #30363D' },
          content: { background: '#0D1117' },
          body: { background: '#0D1117' },
          close: { color: '#8B949E' },
        }}
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <MoveList />
          <SSEEventLog />
          <AnalysisTable />
          <MetricsGrid />
          <RecentGames />
        </div>
      </Drawer>
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
