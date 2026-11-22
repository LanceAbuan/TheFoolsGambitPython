/**
 * Analysis: set up or play any position, then see how the Fool rates it —
 * and how Stockfish disagrees.
 *
 * Position lives in `useAnalysisBoard` (chess.js in play mode, a raw
 * board-map in setup mode). Every position change debounces 600 ms into one
 * POST /api/analyze (both engines), aborted when superseded. Engine arrows
 * overlay the board: emerald = the Fool, orange = Stockfish, deduped when
 * both engines agree. All scores are White-POV, exactly as the API reports.
 */
import { useEffect, useMemo, useState } from 'react';
import {
  Badge,
  Button,
  Card,
  Checkbox,
  Container,
  Group,
  SegmentedControl,
  Select,
  Stack,
  Text,
  TextInput,
  Title,
  Tooltip,
} from '@mantine/core';
import {
  IconCheck,
  IconCopy,
  IconEraser,
  IconRefresh,
  IconArrowBackUp,
  IconSpeakerphone,
} from '@tabler/icons-react';
import { Board, type BoardHint } from '../components/Board';
import {
  analyze,
  type AnalyzeResponse,
  type FoolAnalysis,
  type StockfishAnalysis,
  type TerminalKind,
} from '../lib/api/http';
import { useAnalysisBoard } from '../lib/hooks/useAnalysisBoard';
import { fenEvaluable, type PlacedPiece } from '../lib/analysis/fen';
import { shortCkpt } from '../lib/watch/format';

const FOOL_ARROW = '#2f9e6e';
const STOCKFISH_ARROW = '#e8590c';
const DEBOUNCE_MS = 600;

const TERMINAL_TEXT: Record<TerminalKind, (turn: 'white' | 'black') => string> = {
  checkmate: (turn) => `Checkmate — ${turn === 'white' ? 'Black' : 'White'} wins`,
  stalemate: () => 'Stalemate — draw',
  insufficient: () => 'Draw — insufficient material',
  seventyfive: () => 'Draw — seventy-five-move rule',
  fivefold: () => 'Draw — fivefold repetition',
};

/** White-POV score label: `+0.36`, `−M4`, `—`. Mate 0 (position is mate) shows via the banner. */
function scoreText(cp: number | null, mate: number | null): string {
  if (mate !== null && mate !== 0) return mate > 0 ? `+M${mate}` : `−M${Math.abs(mate)}`;
  if (cp === null || !Number.isFinite(cp)) return '—';
  const v = cp / 100;
  const s = Math.abs(v).toFixed(2);
  return v > 0 ? `+${s}` : v < 0 ? `−${s}` : s;
}

function qText(q: number): string {
  const v = Math.abs(q) < 0.005 ? 0 : q;
  return v > 0 ? `+${v.toFixed(2)}` : v < 0 ? `−${Math.abs(v).toFixed(2)}` : '0.00';
}

/** FEN freshness: compare the fields the server normalizes away (ep can differ). */
function samePosition(a: string | null | undefined, b: string | null | undefined): boolean {
  if (!a || !b) return false;
  const fa = a.trim().split(/\s+/);
  const fb = b.trim().split(/\s+/);
  return fa[0] === fb[0] && fa[1] === fb[1] && fa[2] === fb[2];
}

function ExpectedBar({ value }: { value: number | null }) {
  const w = value === null ? 50 : Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="expected" role="img" aria-label={`White expected score ${Math.round(w)} percent`}>
      <div className="expected-fill" style={{ width: `${w}%` }} />
      <div className="expected-mid" aria-hidden />
    </div>
  );
}

function MoveButton({
  san,
  meta,
  onClick,
}: {
  san: string;
  meta: string;
  onClick: () => void;
}) {
  return (
    <Button
      variant="default"
      size="compact-sm"
      className="top-move"
      justify="space-between"
      fullWidth
      onClick={onClick}
      title={`Play ${san}`}
    >
      <span className="mono top-move-san">{san}</span>
      <span className="top-move-meta">{meta}</span>
    </Button>
  );
}

function EngineUnavailable({ name, error }: { name: string; error: string | null }) {
  return (
    <Stack gap={4} py="xs">
      <Text fz="sm" fw={600}>
        {name} is unavailable
      </Text>
      <Text fz="xs" c="dimmed">
        {error ?? 'No report for this position.'}
      </Text>
    </Stack>
  );
}

function FoolCard({
  engine,
  fresh,
  visits,
  onVisits,
  onPlayMove,
}: {
  engine: FoolAnalysis | undefined;
  fresh: boolean;
  visits: number;
  onVisits: (v: number) => void;
  onPlayMove: (uci: string) => void;
}) {
  const live = engine && engine.available && fresh;
  return (
    <div className={`eval-card${live ? '' : ' eval-dim'}`}>
      <Group justify="space-between" gap="xs" mb={6} wrap="nowrap">
        <Group gap={6} wrap="nowrap">
          <span className="engine-dot fool" aria-hidden />
          <Text fz="sm" fw={600}>
            The Fool
          </Text>
          {engine?.checkpoint ? (
            <Badge variant="light" color="gray" radius="sm" size="xs" className="mono">
              {shortCkpt(engine.checkpoint)}
            </Badge>
          ) : null}
        </Group>
        {live && engine?.elapsed_ms !== null && engine ? (
          <Text fz="xs" c="dimmed" className="mono">
            {(engine.elapsed_ms / 1000).toFixed(1)}s
          </Text>
        ) : null}
      </Group>

      {!engine ? (
        <Text fz="xs" c="dimmed" py="xs">
          No report yet.
        </Text>
      ) : !engine.available ? (
        <EngineUnavailable name="The Fool" error={engine.error} />
      ) : !fresh ? (
        <Text fz="xs" c="dimmed" py="xs">
          Waiting for this position…
        </Text>
      ) : (
        <>
          <Text className="eval-score">{scoreText(engine.score_cp, engine.mate)}</Text>
          <ExpectedBar value={engine.expected_white} />
          <Stack gap={4} mt="sm">
            {engine.top.slice(0, 3).map((m) => (
              <MoveButton
                key={m.uci}
                san={m.san}
                meta={`${m.visits} visits · ${qText(m.q)}`}
                onClick={() => onPlayMove(m.uci)}
              />
            ))}
          </Stack>
        </>
      )}

      <Group justify="space-between" mt="sm" wrap="nowrap">
        <Select
          size="xs"
          w={150}
          aria-label="Fool search budget"
          data={[
            { value: '32', label: '32 · wandering' },
            { value: '96', label: '96 · curious' },
            { value: '256', label: '256 · enlightened' },
          ]}
          value={String(visits)}
          onChange={(v) => v && onVisits(Number(v))}
          allowDeselect={false}
        />
        {live && engine?.pv.length ? (
          <Text fz={11} c="dimmed" className="mono" lineClamp={1} style={{ minWidth: 0 }}>
            {engine.pv.join(' ')}
          </Text>
        ) : null}
      </Group>
    </div>
  );
}

function StockfishCard({
  engine,
  fresh,
  depth,
  onDepth,
  onPlayMove,
}: {
  engine: StockfishAnalysis | undefined;
  fresh: boolean;
  depth: number;
  onDepth: (v: number) => void;
  onPlayMove: (uci: string) => void;
}) {
  const live = engine && engine.available && fresh;
  return (
    <div className={`eval-card${live ? '' : ' eval-dim'}`}>
      <Group justify="space-between" gap="xs" mb={6} wrap="nowrap">
        <Group gap={6} wrap="nowrap">
          <span className="engine-dot stockfish" aria-hidden />
          <Text fz="sm" fw={600}>
            Stockfish
          </Text>
          {live && engine?.depth ? (
            <Badge variant="light" color="gray" radius="sm" size="xs" className="mono">
              depth {engine.depth}
            </Badge>
          ) : null}
        </Group>
        {live && engine?.elapsed_ms !== null && engine ? (
          <Text fz="xs" c="dimmed" className="mono">
            {(engine.elapsed_ms / 1000).toFixed(1)}s
          </Text>
        ) : null}
      </Group>

      {!engine ? (
        <Text fz="xs" c="dimmed" py="xs">
          No report yet.
        </Text>
      ) : !engine.available ? (
        <EngineUnavailable name="Stockfish" error={engine.error} />
      ) : !fresh ? (
        <Text fz="xs" c="dimmed" py="xs">
          Waiting for this position…
        </Text>
      ) : (
        <>
          <Text className="eval-score">{scoreText(engine.score_cp, engine.mate)}</Text>
          <ExpectedBar value={engine.expected_white} />
          <Stack gap={4} mt="sm">
            {engine.top.slice(0, 3).map((m) => (
              <MoveButton
                key={m.uci}
                san={m.san ?? m.uci}
                meta={scoreText(m.score_cp, m.mate)}
                onClick={() => onPlayMove(m.uci)}
              />
            ))}
          </Stack>
        </>
      )}

      <Group justify="space-between" mt="sm" wrap="nowrap">
        <Select
          size="xs"
          w={110}
          aria-label="Stockfish depth"
          data={['10', '14', '16', '18', '22'].map((d) => ({ value: d, label: `depth ${d}` }))}
          value={String(depth)}
          onChange={(v) => v && onDepth(Number(v))}
          allowDeselect={false}
        />
        {live && engine?.pv.length ? (
          <Text fz={11} c="dimmed" className="mono" lineClamp={1} style={{ minWidth: 0 }}>
            {engine.pv.join(' ')}
          </Text>
        ) : null}
      </Group>
    </div>
  );
}

const PALETTE: { key: string; piece: PlacedPiece | null; label: string }[] = [
  { key: 'wk', piece: { type: 'k', color: 'w' }, label: 'white king' },
  { key: 'wq', piece: { type: 'q', color: 'w' }, label: 'white queen' },
  { key: 'wr', piece: { type: 'r', color: 'w' }, label: 'white rook' },
  { key: 'wb', piece: { type: 'b', color: 'w' }, label: 'white bishop' },
  { key: 'wn', piece: { type: 'n', color: 'w' }, label: 'white knight' },
  { key: 'wp', piece: { type: 'p', color: 'w' }, label: 'white pawn' },
  { key: 'bk', piece: { type: 'k', color: 'b' }, label: 'black king' },
  { key: 'bq', piece: { type: 'q', color: 'b' }, label: 'black queen' },
  { key: 'br', piece: { type: 'r', color: 'b' }, label: 'black rook' },
  { key: 'bb', piece: { type: 'b', color: 'b' }, label: 'black bishop' },
  { key: 'bn', piece: { type: 'n', color: 'b' }, label: 'black knight' },
  { key: 'bp', piece: { type: 'p', color: 'b' }, label: 'black pawn' },
  { key: 'eraser', piece: null, label: 'eraser' },
];

export function Analysis() {
  const board = useAnalysisBoard();
  const [orientation, setOrientation] = useState<'white' | 'black'>('white');
  const [brush, setBrush] = useState('wp');
  const [visits, setVisits] = useState(96);
  const [depth, setDepth] = useState(16);
  const [report, setReport] = useState<AnalyzeResponse | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [fenInput, setFenInput] = useState(board.fen);

  const { mode } = board;
  const evaluable = mode === 'setup' ? board.editorEvaluable : fenEvaluable(board.fen);
  const fresh = samePosition(report?.fen, board.fen);

  useEffect(() => setFenInput(board.fen), [board.fen]);

  // Debounced, abort-aware analysis. A superseded request never lands.
  useEffect(() => {
    if (!evaluable) {
      setReport(null);
      setAnalyzing(false);
      setError(null);
      return;
    }
    let alive = true;
    setAnalyzing(true);
    const ctrl = new AbortController();
    const timer = setTimeout(() => {
      analyze({ fen: board.fen, engines: ['fool', 'stockfish'], visits, depth, multipv: 3 }, ctrl.signal)
        .then((r) => {
          if (!alive) return;
          setReport(r);
          setError(null);
          setAnalyzing(false);
        })
        .catch((e: unknown) => {
          if (!alive || (e as Error)?.name === 'AbortError') return;
          setError((e as Error)?.message ?? 'analysis request failed');
          setAnalyzing(false);
        });
    }, DEBOUNCE_MS);
    return () => {
      alive = false;
      clearTimeout(timer);
      ctrl.abort();
    };
  }, [board.fen, evaluable, visits, depth]);

  const hints = useMemo<BoardHint[]>(() => {
    if (!fresh || report?.terminal) return [];
    const fool = report?.engines.fool;
    const sf = report?.engines.stockfish;
    const out: BoardHint[] = [];
    if (fool?.available && fool.best_move) {
      out.push({ from: fool.best_move.slice(0, 2), to: fool.best_move.slice(2, 4), color: FOOL_ARROW });
    }
    if (sf?.available && sf.best_move && sf.best_move !== fool?.best_move) {
      out.push({ from: sf.best_move.slice(0, 2), to: sf.best_move.slice(2, 4), color: STOCKFISH_ARROW });
    }
    return out;
  }, [fresh, report]);

  const editorClick = (sq: string): void => {
    if (brush === 'eraser') {
      board.editSquare(sq, null);
      return;
    }
    const picked = PALETTE.find((p) => p.key === brush);
    if (!picked?.piece) return;
    const existing = board.editor.map[sq];
    if (existing && existing.type === picked.piece.type && existing.color === picked.piece.color) {
      board.editSquare(sq, null); // same piece again → erase
    } else {
      board.editSquare(sq, picked.piece);
    }
  };

  const copyFen = (): void => {
    navigator.clipboard?.writeText(board.fen).then(
      () => {
        setCopied(true);
        setTimeout(() => setCopied(false), 1200);
      },
      () => {},
    );
  };

  const turnLabel = board.fen.split(/\s+/)[1] === 'b' ? 'Black to move' : 'White to move';
  const movePairs = useMemo(() => {
    const out: { no: number; w?: string; b?: string }[] = [];
    board.sanHistory.forEach((san, i) => {
      const no = Math.floor(i / 2) + 1;
      if (i % 2 === 0) out.push({ no, w: san });
      else {
        const pair = out[out.length - 1];
        if (pair && pair.no === no) pair.b = san;
        else out.push({ no, b: san });
      }
    });
    return out;
  }, [board.sanHistory]);

  return (
    <Container size={1180} py={{ base: 24, sm: 32 }}>
      <Stack gap="lg">
        <Group justify="space-between" align="flex-end" gap="md" wrap="nowrap">
          <Title order={2}>Analyze</Title>
          <Badge
            variant="light"
            color={report?.terminal ? 'orange' : 'gray'}
            radius="sm"
            size="md"
            className="mono"
          >
            {report?.terminal && fresh ? TERMINAL_TEXT[report.terminal](report.turn) : turnLabel}
          </Badge>
        </Group>

        <div className="board-layout">
          <Card padding="md" className="board-card">
            <Board
              fen={board.fen}
              lastMove={board.lastMove}
              checkSquare={board.checkSquare}
              orientation={orientation}
              legalTargets={board.legalTargets}
              onMove={(from, to, promo) => board.tryMove(from, to, promo)}
              onSquareClick={mode === 'setup' ? editorClick : undefined}
              hints={hints}
            />
            <Group justify="space-between" gap="xs" mt="sm" wrap="nowrap">
              <Group gap={4} wrap="nowrap" className="move-strip">
                {mode === 'play' ? (
                  movePairs.length === 0 ? (
                    <Text fz="xs" c="dimmed">
                      Play a move, or switch to Set up to place pieces freely.
                    </Text>
                  ) : (
                    movePairs.slice(-6).map((p) => (
                      <Text key={p.no} fz="xs" className="mono" c="dimmed" style={{ whiteSpace: 'nowrap' }}>
                        {p.no}.{p.w ? ` ${p.w}` : ''}
                        {p.b ? ` ${p.b}` : ''}
                      </Text>
                    ))
                  )
                ) : (
                  <Text fz="xs" c="dimmed">
                    Pick a piece, then click squares to place it. Click the same piece to erase it.
                  </Text>
                )}
              </Group>
              <Tooltip label="Flip board">
                <Button
                  variant="default"
                  size="compact-sm"
                  aria-label="Flip board"
                  onClick={() => setOrientation((o) => (o === 'white' ? 'black' : 'white'))}
                >
                  Flip
                </Button>
              </Tooltip>
            </Group>
          </Card>

          <Stack gap="md" className="rail">
            <Card padding="md">
              <SegmentedControl
                fullWidth
                value={mode}
                onChange={(v) => (v === 'setup' ? board.enterSetup() : board.leaveSetup())}
                data={[
                  { value: 'play', label: 'Play moves' },
                  { value: 'setup', label: 'Set up position' },
                ]}
              />

              {board.notice ? (
                <Text fz={11.5} c="orange" mt="sm">
                  {board.notice}
                </Text>
              ) : null}

              {mode === 'play' ? (
                <>
                  <Group gap="xs" mt="sm" wrap="nowrap">
                    <Button
                      variant="default"
                      size="compact-sm"
                      leftSection={<IconArrowBackUp size={14} />}
                      onClick={board.undo}
                      disabled={board.sanHistory.length === 0}
                    >
                      Undo
                    </Button>
                    <Button
                      variant="default"
                      size="compact-sm"
                      leftSection={<IconRefresh size={14} />}
                      onClick={board.reset}
                    >
                      Reset
                    </Button>
                  </Group>
                  <Group gap={6} mt="sm" wrap="nowrap" align="flex-start">
                    <TextInput
                      size="xs"
                      flex={1}
                      aria-label="FEN"
                      value={fenInput}
                      onChange={(e) => setFenInput(e.currentTarget.value)}
                      error={fenInput.trim() !== board.fen ? undefined : undefined}
                      styles={{ input: { textOverflow: 'ellipsis' } }}
                    />
                    <Button
                      variant="default"
                      size="compact-xs"
                      onClick={() => {
                        if (!board.loadFen(fenInput)) board.setNotice('That FEN could not be loaded.');
                      }}
                    >
                      Load
                    </Button>
                    <Button
                      variant="default"
                      size="compact-xs"
                      aria-label="Copy FEN"
                      leftSection={copied ? <IconCheck size={13} /> : <IconCopy size={13} />}
                      onClick={copyFen}
                    >
                      {copied ? 'Copied' : 'Copy'}
                    </Button>
                  </Group>
                </>
              ) : (
                <>
                  <div className="palette" role="toolbar" aria-label="Piece palette">
                    {PALETTE.map((p) => (
                      <button
                        key={p.key}
                        type="button"
                        className={`palette-btn${brush === p.key ? ' selected' : ''}`}
                        aria-pressed={brush === p.key}
                        aria-label={p.label}
                        title={p.label}
                        onClick={() => setBrush(p.key)}
                      >
                        {p.piece ? (
                          <img src={`/pieces/${p.piece.color}${p.piece.type.toUpperCase()}.svg`} alt="" />
                        ) : (
                          <IconEraser size={16} />
                        )}
                      </button>
                    ))}
                  </div>

                  <Group gap="xs" mt="sm" wrap="nowrap">
                    <SegmentedControl
                      size="xs"
                      flex={1}
                      aria-label="Side to move"
                      value={board.editor.meta.turn}
                      onChange={(v) => board.editMeta({ turn: v as 'w' | 'b' })}
                      data={[
                        { value: 'w', label: 'White to move' },
                        { value: 'b', label: 'Black to move' },
                      ]}
                    />
                  </Group>

                  <Group gap="md" mt="sm" wrap="nowrap">
                    <Checkbox
                      size="xs"
                      label="White O-O"
                      disabled={board.editor.meta.castling.includes('K') ? false : !board.editor.map.h1 || !board.editor.map.e1}
                      checked={board.editor.meta.castling.includes('K')}
                      onChange={(e) =>
                        board.editMeta({
                          castling: e.currentTarget.checked
                            ? board.editor.meta.castling.replace('-', '') + 'K'
                            : board.editor.meta.castling.replace('K', '') || '-',
                        })
                      }
                    />
                    <Checkbox
                      size="xs"
                      label="White O-O-O"
                      disabled={board.editor.meta.castling.includes('Q') ? false : !board.editor.map.a1 || !board.editor.map.e1}
                      checked={board.editor.meta.castling.includes('Q')}
                      onChange={(e) =>
                        board.editMeta({
                          castling: e.currentTarget.checked
                            ? board.editor.meta.castling.replace('-', '') + 'Q'
                            : board.editor.meta.castling.replace('Q', '') || '-',
                        })
                      }
                    />
                  </Group>
                  <Group gap="md" mt={6} wrap="nowrap">
                    <Checkbox
                      size="xs"
                      label="Black O-O"
                      disabled={board.editor.meta.castling.includes('k') ? false : !board.editor.map.h8 || !board.editor.map.e8}
                      checked={board.editor.meta.castling.includes('k')}
                      onChange={(e) =>
                        board.editMeta({
                          castling: e.currentTarget.checked
                            ? board.editor.meta.castling.replace('-', '') + 'k'
                            : board.editor.meta.castling.replace('k', '') || '-',
                        })
                      }
                    />
                    <Checkbox
                      size="xs"
                      label="Black O-O-O"
                      disabled={board.editor.meta.castling.includes('q') ? false : !board.editor.map.a8 || !board.editor.map.e8}
                      checked={board.editor.meta.castling.includes('q')}
                      onChange={(e) =>
                        board.editMeta({
                          castling: e.currentTarget.checked
                            ? board.editor.meta.castling.replace('-', '') + 'q'
                            : board.editor.meta.castling.replace('q', '') || '-',
                        })
                      }
                    />
                  </Group>

                  <TextInput
                    size="xs"
                    mt="sm"
                    label="En passant target"
                    placeholder="— (none)"
                    value={board.editor.meta.ep === '-' ? '' : board.editor.meta.ep}
                    onChange={(e) => {
                      const v = e.currentTarget.value.trim().toLowerCase();
                      board.editMeta({ ep: v === '' ? '-' : v });
                    }}
                    error={board.editor.meta.ep !== '-' && !/^[a-h][36]$/.test(board.editor.meta.ep) ? 'a3–h6' : undefined}
                  />

                  <Group gap="xs" mt="sm" wrap="nowrap">
                    <Button variant="default" size="compact-sm" onClick={board.clearBoard}>
                      Clear board
                    </Button>
                    <Button
                      variant="filled"
                      size="compact-sm"
                      onClick={board.leaveSetup}
                      title={
                        board.editorEvaluable
                          ? 'Return to play mode with this position'
                          : 'Both kings are required before this can be played'
                      }
                    >
                      Play from here
                    </Button>
                  </Group>

                  <Group gap={6} mt="sm" wrap="nowrap" align="flex-start">
                    <TextInput
                      size="xs"
                      flex={1}
                      aria-label="FEN"
                      value={fenInput}
                      onChange={(e) => setFenInput(e.currentTarget.value)}
                      styles={{ input: { textOverflow: 'ellipsis' } }}
                    />
                    <Button
                      variant="default"
                      size="compact-xs"
                      onClick={() => {
                        if (!board.loadEditorFen(fenInput)) board.setNotice('That FEN could not be parsed.');
                      }}
                    >
                      Load
                    </Button>
                    <Button
                      variant="default"
                      size="compact-xs"
                      aria-label="Copy FEN"
                      leftSection={copied ? <IconCheck size={13} /> : <IconCopy size={13} />}
                      onClick={copyFen}
                    >
                      {copied ? 'Copied' : 'Copy'}
                    </Button>
                  </Group>
                </>
              )}
            </Card>

            <Card padding="md">
              <Group justify="space-between" gap="xs" mb="sm">
                <Text fw={600}>Evaluation</Text>
                <Text fz="xs" c="dimmed" className="mono">
                  {analyzing ? 'Analyzing…' : fresh ? 'Up to date' : evaluable ? 'Queued' : 'Place both kings'}
                </Text>
              </Group>

              {error ? (
                <Text fz="xs" c="orange" mb="sm">
                  {error}
                </Text>
              ) : null}
              {!evaluable ? (
                <Text fz="xs" c="dimmed" mb="sm">
                  A position needs both kings before it can be evaluated. Place them in Set up mode.
                </Text>
              ) : null}
              {report?.terminal && fresh ? (
                <div className="terminal-banner">
                  <IconSpeakerphone size={15} aria-hidden />
                  <Text fz="sm" fw={600}>
                    {TERMINAL_TEXT[report.terminal](report.turn)}
                  </Text>
                </div>
              ) : null}

              <FoolCard
                engine={report?.engines.fool}
                fresh={fresh}
                visits={visits}
                onVisits={setVisits}
                onPlayMove={board.applyUci}
              />
              <div className="eval-divider" aria-hidden />
              <StockfishCard
                engine={report?.engines.stockfish}
                fresh={fresh}
                depth={depth}
                onDepth={setDepth}
                onPlayMove={board.applyUci}
              />

              <Text fz="xs" c="dimmed" mt="sm">
                The Fool runs a short Monte-Carlo search on this machine&apos;s GPU; Stockfish a classic
                alpha-beta search. Scores are from White&apos;s point of view. Click a move to play it.
              </Text>
            </Card>
          </Stack>
        </div>
      </Stack>
    </Container>
  );
}
