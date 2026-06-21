import { createTheme } from '@mantine/core';

export const theme = createTheme({
  fontFamily: '"Inter", system-ui, -apple-system, sans-serif',
  fontFamilyMonospace: 'ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace',
  primaryColor: 'blue',
  primaryShade: 6,
  defaultRadius: 'sm',

  colors: {
    // GitHub-dark inspired dark palette — used everywhere as background/base
    dark: [
      '#C9D1D9', // 0  - primary text on dark
      '#8B949E', // 1  - secondary text
      '#6E7681', // 2  - muted / tertiary text
      '#484F58', // 3  - disabled / icon
      '#30363D', // 4  - border color
      '#21262D', // 5  - hover state / subtle bg
      '#161B22', // 6  - card / surface background
      '#0D1117', // 7  - page background
      '#0A0E14', // 8  - deeper bg
      '#06090E', // 9  - deepest bg
    ],

    // Blue accent (primary) — proper 10-shade array for Mantine's color system
    blue: [
      '#d4e6ff',
      '#adccf8',
      '#84b2f0',
      '#5c98e8',
      '#3a80e0',
      '#2d6bc4', // 5 - base
      '#2358a8',
      '#1a468c',
      '#123470',
      '#0a2355',
    ],

    // Green accent — for positive metrics, best moves
    green: [
      '#d3ffd3',
      '#aff0af',
      '#8ae08a',
      '#62d062',
      '#4cc04c',
      '#3fb950', // 5 - base
      '#32a040',
      '#268833',
      '#1a7026',
      '#0e5819',
    ],

    // Orange accent — for warnings, bad moves
    orange: [
      '#ffe0c2',
      '#fcc89a',
      '#f8af70',
      '#f49646',
      '#f07e1e',
      '#d66818', // 5 - base
      '#bc5312',
      '#a23f0c',
      '#882c06',
      '#6e1a00',
    ],

    // Red accent — for errors, blunders
    red: [
      '#ffd6d4',
      '#fcaeb0',
      '#f8868c',
      '#f45e68',
      '#f03644',
      '#d62d3a', // 5 - base
      '#bc2430',
      '#a21b26',
      '#88121c',
      '#6e0912',
    ],

    // Yellow accent — for warnings, connecting state
    yellow: [
      '#fff8d4',
      '#fceea8',
      '#f8e47c',
      '#f4da50',
      '#f0d024',
      '#d6b81e', // 5 - base
      '#bca018',
      '#a28812',
      '#88700c',
      '#6e5806',
    ],

    // Purple / violet — for book moves, special tags
    grape: [
      '#f0d4ff',
      '#dca8f8',
      '#c87cf0',
      '#b450e8',
      '#a024e0',
      '#8c1ec4', // 5 - base
      '#7818a8',
      '#64128c',
      '#500c70',
      '#3c0655',
    ],

    // Chess move quality colors (as named tokens)
    'move-best': ['#3fb950', '#3fb950', '#3fb950', '#3fb950', '#3fb950', '#3fb950', '#3fb950', '#3fb950', '#3fb950', '#3fb950'],
    'move-good': ['#58a6ff', '#58a6ff', '#58a6ff', '#58a6ff', '#58a6ff', '#58a6ff', '#58a6ff', '#58a6ff', '#58a6ff', '#58a6ff'],
    'move-ok': ['#8b949e', '#8b949e', '#8b949e', '#8b949e', '#8b949e', '#8b949e', '#8b949e', '#8b949e', '#8b949e', '#8b949e'],
    'move-bad': ['#f0883e', '#f0883e', '#f0883e', '#f0883e', '#f0883e', '#f0883e', '#f0883e', '#f0883e', '#f0883e', '#f0883e'],
    'move-blunder': ['#f85149', '#f85149', '#f85149', '#f85149', '#f85149', '#f85149', '#f85149', '#f85149', '#f85149', '#f85149'],
    'move-book': ['#bc8cff', '#bc8cff', '#bc8cff', '#bc8cff', '#bc8cff', '#bc8cff', '#bc8cff', '#bc8cff', '#bc8cff', '#bc8cff'],
  },

  components: {
    Paper: {
      defaultProps: {
        p: 'md',
        radius: 'md',
      },
    },
    Table: {
      defaultProps: {
        striped: true,
        highlightOnHover: true,
        withTableBorder: false,
        withColumnBorders: false,
      },
    },
    Badge: {
      defaultProps: { size: 'sm' },
    },
    Tooltip: {
      defaultProps: { openDelay: 500 },
    },
  },

  other: {
    colors: {
      textPrimary: '#C9D1D9',
      textSecondary: '#8B949E',
      textMuted: '#6E7681',
      borderColor: '#30363D',
      cardBg: '#161B22',
      inputBg: '#0D1117',
      hoverBg: '#21262D',
      chessDark: '#625b4d',
      chessLight: '#b7b09c',
      sectionHeaderBorder: '#30363D',
      accentBlue: '#58a6ff',
      // Tab colors
      tabActive: '#58a6ff',
      tabInactive: '#8B949E',
      tabHover: 'rgba(88, 166, 255, 0.08)',
      // Status colors
      statusRunning: '#3fb950',
      statusThinking: '#d6b81e',
      statusFinished: '#8B949E',
      statusWaiting: '#58a6ff',
      // Log level colors
      logInfo: '#58a6ff',
      logWarning: '#d6b81e',
      logError: '#f85149',
    },
  },
});
