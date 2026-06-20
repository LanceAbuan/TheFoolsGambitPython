import { useEffect } from 'react';
import { Routes, Route } from 'react-router-dom';
import { AppShell } from '@mantine/core';
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
        <TopBar />
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

          {/* Side games column — 3×3 grid between board and sidebar */}
          <div className="side-games-column">
            <div className="side-boards-grid">
              {[1, 2, 3, 4, 5, 6, 7, 8, 9].map((id) => (
                <SideBoard key={id} gameId={id} />
              ))}
            </div>
          </div>

          {/* Right sidebar — stacked layout */}
          <div className="sidebar-column">
            <MoveList />
            <SSEEventLog />
            <AnalysisTable />
            <MetricsGrid />
            <RecentGames />
          </div>
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
