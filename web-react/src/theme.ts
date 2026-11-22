import {
  Button,
  Card,
  createTheme,
  Paper,
  type MantineColorsTuple,
} from '@mantine/core';

// Fresh identity for the React client: one deep-emerald accent on Mantine's
// neutral ladder. No carry-over from the Svelte palette — this is a new design,
// not a port. Dark mode is first-class (auto by default, toggle in the header).
const emerald: MantineColorsTuple = [
  '#e9f6ef',
  '#d2e9de',
  '#a5d2bd',
  '#74b99a',
  '#4ea37d',
  '#37956b',
  '#2a8c61', // primary (light)
  '#1f7250',
  '#15573d',
  '#0a3a28',
];

export const theme = createTheme({
  primaryColor: 'emerald',
  primaryShade: { light: 6, dark: 4 },
  colors: { emerald },
  fontFamily:
    'Inter, "Segoe UI Variable Text", "Segoe UI", system-ui, -apple-system, sans-serif',
  fontFamilyMonospace:
    'ui-monospace, "Cascadia Code", "SFMono-Regular", Consolas, monospace',
  headings: {
    fontFamily:
      'Inter, "Segoe UI Variable Display", "Segoe UI", system-ui, -apple-system, sans-serif',
    fontWeight: '650',
    sizes: {
      h1: { fontSize: 'clamp(2.5rem, 6vw, 4.25rem)', lineHeight: '1.02' },
      h2: { fontSize: 'clamp(1.6rem, 3vw, 2.1rem)', lineHeight: '1.15' },
      h3: { fontSize: '1.15rem', lineHeight: '1.25' },
    },
  },
  defaultRadius: 'md',
  cursorType: 'pointer',
  components: {
    Card: Card.extend({
      defaultProps: { radius: 'lg', padding: 'lg', withBorder: true, shadow: 'sm' },
    }),
    Paper: Paper.extend({ defaultProps: { radius: 'lg' } }),
    Button: Button.extend({ defaultProps: { radius: 'md' } }),
  },
});
