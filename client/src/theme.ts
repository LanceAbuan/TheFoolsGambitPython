import { createTheme } from '@mantine/core';

export const theme = createTheme({
  fontFamily: '"Inter", system-ui, -apple-system, sans-serif',
  fontFamilyMonospace: 'ui-monospace, Consolas, monospace',
  primaryColor: 'accent-green',
  defaultRadius: 'sm',
  colors: {
    'bg-primary': ['#312e22', '#312e22', '#312e22', '#312e22', '#312e22', '#312e22', '#312e22', '#312e22', '#312e22', '#312e22'],
    'bg-secondary': ['#272522', '#272522', '#272522', '#272522', '#272522', '#272522', '#272522', '#272522', '#272522', '#272522'],
    'bg-tertiary': ['#211f1c', '#211f1c', '#211f1c', '#211f1c', '#211f1c', '#211f1c', '#211f1c', '#211f1c', '#211f1c', '#211f1c'],
    'bg-hover': ['#3d3935', '#3d3935', '#3d3935', '#3d3935', '#3d3935', '#3d3935', '#3d3935', '#3d3935', '#3d3935', '#3d3935'],
    'accent-green': ['#81b64c', '#81b64c', '#81b64c', '#81b64c', '#81b64c', '#81b64c', '#81b64c', '#81b64c', '#81b64c', '#81b64c'],
    'accent-blue': ['#4a90d9', '#4a90d9', '#4a90d9', '#4a90d9', '#4a90d9', '#4a90d9', '#4a90d9', '#4a90d9', '#4a90d9', '#4a90d9'],
    'accent-orange': ['#e88734', '#e88734', '#e88734', '#e88734', '#e88734', '#e88734', '#e88734', '#e88734', '#e88734', '#e88734'],
    'accent-red': ['#e94560', '#e94560', '#e94560', '#e94560', '#e94560', '#e94560', '#e94560', '#e94560', '#e94560', '#e94560'],
    'accent-yellow': ['#f0c040', '#f0c040', '#f0c040', '#f0c040', '#f0c040', '#f0c040', '#f0c040', '#f0c040', '#f0c040', '#f0c040'],
    'move-best': ['#81b64c', '#81b64c', '#81b64c', '#81b64c', '#81b64c', '#81b64c', '#81b64c', '#81b64c', '#81b64c', '#81b64c'],
    'move-good': ['#4a90d9', '#4a90d9', '#4a90d9', '#4a90d9', '#4a90d9', '#4a90d9', '#4a90d9', '#4a90d9', '#4a90d9', '#4a90d9'],
    'move-ok': ['#8a8580', '#8a8580', '#8a8580', '#8a8580', '#8a8580', '#8a8580', '#8a8580', '#8a8580', '#8a8580', '#8a8580'],
    'move-bad': ['#e88734', '#e88734', '#e88734', '#e88734', '#e88734', '#e88734', '#e88734', '#e88734', '#e88734', '#e88734'],
    'move-blunder': ['#e94560', '#e94560', '#e94560', '#e94560', '#e94560', '#e94560', '#e94560', '#e94560', '#e94560', '#e94560'],
    'move-book': ['#9b59b6', '#9b59b6', '#9b59b6', '#9b59b6', '#9b59b6', '#9b59b6', '#9b59b6', '#9b59b6', '#9b59b6', '#9b59b6'],
  },
  components: {
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
    ActionIcon: {
      defaultProps: { variant: 'subtle' },
    },
  },
  other: {
    colors: {
      textPrimary: '#fff',
      textSecondary: '#b0a99f',
      textMuted: '#6b6560',
      borderColor: '#3d3935',
    },
  },
});
