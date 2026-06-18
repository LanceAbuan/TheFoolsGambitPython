import { useEffect } from 'react';
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

export default function App() {
  const { state } = useGame();

  // Initialize hooks
  useSSE();
  useStatus();
  useAnalysis();

  useEffect(() => {
    // Update autoFollow when new moves come in
    if (state.autoFollow && state.allMoves.length > 0) {
      // The view index is already updated in the SSE hook
    }
  }, [state.allMoves.length, state.autoFollow]);

  // Player turn detection
  const whiteToMove = state.allMoves.length % 2 === 0;

  return (
    <>
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

              {/* Side boards grid */}
              <div className="side-boards-grid">
                {[2, 3, 4, 5, 6, 7, 8, 9, 10].map((id) => (
                  <SideBoard key={id} gameId={id} />
                ))}
              </div>
            </div>

            {/* Right sidebar — cube/grid layout */}
            <div className="sidebar-column">
              <div><MoveList /></div>
              <div><SSEEventLog /></div>
              <div className="sidebar-full"><AnalysisTable /></div>
              <div className="sidebar-full"><MetricsGrid /></div>
              <div className="sidebar-full"><RecentGames /></div>
            </div>
          </div>
        </AppShell.Main>
      </AppShell>

      <FullscreenOverlay />
    </>
  );
}
