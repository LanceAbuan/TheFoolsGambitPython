import { useEffect } from 'react';
import { Routes, Route } from 'react-router-dom';
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
    <div className="dashboard-layout">
      <div className="main-column">
        {/* Header scoped to main board area */}
        <TopBar />

        <PlayerInfoBar
          name="Self-Play (NN)"
          detail="vs Self"
          isTurn={!whiteToMove}
          color="top"
        />

        {/* Board + side games side by side */}
        <div className="board-and-sides">
          <div className="board-row">
            <EvalBar />
            <LiveBoard />
          </div>
          <div className="side-boards-grid">
            {[1, 2, 3, 4, 5, 6, 7, 8, 9].map((id) => (
              <SideBoard key={id} gameId={id} />
            ))}
          </div>
        </div>

        <PlayerInfoBar
          name="Fool's Gambit AI"
          detail="Training"
          isTurn={whiteToMove}
          color="bottom"
        />
        <BoardNav />
      </div>

      {/* Fixed sidebar */}
      <div className="sidebar-column">
        <MoveList />
        <SSEEventLog />
        <AnalysisTable />
        <MetricsGrid />
        <RecentGames />
      </div>
    </div>
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
