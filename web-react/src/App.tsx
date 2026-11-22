import { AppShell, Box, Container, Group, Text, UnstyledButton } from '@mantine/core';
import { NavLink, Route, Routes } from 'react-router-dom';
import { ColorSchemeToggle } from './components/ColorSchemeToggle';
import { Knight } from './components/Knight';
import { Home } from './pages/Home';
import { Play } from './pages/Play';
import { Analysis } from './pages/Analysis';
import { Watch } from './pages/Watch';
import { About } from './pages/About';

const NAV = [
  { to: '/', label: 'Home', end: true },
  { to: '/play', label: 'Play', end: false },
  { to: '/analysis', label: 'Analyze', end: false },
  { to: '/watch', label: 'Watch', end: false },
  { to: '/about', label: 'About', end: false },
];

export function App() {
  return (
    <AppShell header={{ height: 62 }} padding={0}>
      <AppShell.Header className="site-head">
        <Container size={1180} h="100%">
          <Group h="100%" justify="space-between" wrap="nowrap" gap={8}>
            <UnstyledButton
              component={NavLink}
              to="/"
              aria-label="The Fool — home"
              className="brand"
            >
              <Group gap={10} wrap="nowrap">
                <Knight size={22} />
                <Text fw={700} fz="sm" lts="0.16em" visibleFrom="xs" lh={1}>
                  THE FOOL
                </Text>
              </Group>
            </UnstyledButton>

            <Group gap={2} wrap="nowrap">
              {NAV.map((item) => (
                <NavLink key={item.to} to={item.to} end={item.end} className="nav-link">
                  {item.label}
                </NavLink>
              ))}
            </Group>

            <ColorSchemeToggle />
          </Group>
        </Container>
      </AppShell.Header>

      <AppShell.Main>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/play" element={<Play />} />
          <Route path="/analysis" element={<Analysis />} />
          <Route path="/watch" element={<Watch />} />
          <Route path="/about" element={<About />} />
        </Routes>

        {/* Static footer: flows after the content, so it can never overpaint a
            card on a short mobile viewport (the fixed variant did exactly that). */}
        <Box component="footer" className="site-foot">
          <Container size={1180}>
            <Group h={52} justify="space-between" gap="xs" wrap="nowrap">
              <Text fz={10} c="dimmed" lts="0.12em" truncate>
                THE FOOL · A CHESS EXPERIMENT
              </Text>
              <Text fz={10} c="dimmed" className="mono" visibleFrom="xs" truncate>
                local-first · RTX 3080
              </Text>
            </Group>
          </Container>
        </Box>
      </AppShell.Main>
    </AppShell>
  );
}
