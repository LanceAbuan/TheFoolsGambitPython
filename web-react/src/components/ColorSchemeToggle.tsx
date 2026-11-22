import { ActionIcon, Tooltip, useComputedColorScheme, useMantineColorScheme } from '@mantine/core';
import { IconMoon, IconSun } from '@tabler/icons-react';

export function ColorSchemeToggle() {
  const { setColorScheme } = useMantineColorScheme();
  // useComputedColorScheme resolves 'auto' against the OS preference, so the icon
  // matches what is actually rendered even when the scheme was never set explicitly.
  const computed = useComputedColorScheme('light', { getInitialValueInEffect: true });
  const dark = computed === 'dark';
  return (
    <Tooltip label={dark ? 'Switch to light mode' : 'Switch to dark mode'} withArrow>
      <ActionIcon
        variant="default"
        size="lg"
        radius="md"
        aria-label="Toggle colour scheme"
        onClick={() => setColorScheme(dark ? 'light' : 'dark')}
      >
        {dark ? <IconSun size={18} stroke={1.7} /> : <IconMoon size={18} stroke={1.7} />}
      </ActionIcon>
    </Tooltip>
  );
}
