import {
  Button,
  Card,
  createTheme,
  Paper,
  Tooltip,
  type MantineColorsTuple,
} from '@mantine/core';

// One deep-emerald accent on Mantine's neutral ladder. Type scale is
// 12/14/16/20/28 at weights 400/600; cards are flat (1px border, no shadow).
// Dark mode is first-class (auto by default, toggle in the header).
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
    fontWeight: '600',
    sizes: {
      h1: { fontSize: '28px', lineHeight: '1.2' },
      h2: { fontSize: '20px', lineHeight: '1.3' },
      h3: { fontSize: '16px', lineHeight: '1.35' },
      h4: { fontSize: '14px', lineHeight: '1.4' },
    },
  },
  defaultRadius: 'md',
  cursorType: 'pointer',
  components: {
    Card: Card.extend({
      defaultProps: { radius: 'md', padding: 'lg', withBorder: true, shadow: '0' },
    }),
    Paper: Paper.extend({ defaultProps: { radius: 'md' } }),
    Button: Button.extend({ defaultProps: { radius: 'md' } }),
    // Mantine 9 inverts tooltips in dark scheme (near-white bg, black text, black
    // border) — reads as a glitch on dark pages. Force one dark tooltip style in
    // both color schemes: dark-7 fill, light text, hairline border. Arrow inherits.
    Tooltip: Tooltip.extend({
      styles: {
        tooltip: {
          backgroundColor: 'var(--mantine-color-dark-7)',
          color: 'var(--mantine-color-gray-2)',
          border: '1px solid var(--mantine-color-dark-5)',
        },
      },
    }),
  },
});
