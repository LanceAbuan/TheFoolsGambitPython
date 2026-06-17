import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import '@mantine/core/styles.css';
import '@mantine/notifications/styles.css';
import './styles/global.css';
import { MantineProvider } from '@mantine/core';
import { theme } from './theme';
import App from './App';
import { GameProvider } from './GameContext';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <MantineProvider theme={theme} defaultColorScheme="dark">
      <GameProvider>
        <App />
      </GameProvider>
    </MantineProvider>
  </StrictMode>
);
