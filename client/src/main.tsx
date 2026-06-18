import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import '@mantine/core/styles.css';
import '@mantine/notifications/styles.css';
import './styles/global.css';
import { MantineProvider } from '@mantine/core';
import { theme } from './theme';
import App from './App';
import { GameProvider } from './GameContext';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <MantineProvider theme={theme} defaultColorScheme="dark">
        <GameProvider>
          <App />
        </GameProvider>
      </MantineProvider>
    </BrowserRouter>
  </StrictMode>
);
