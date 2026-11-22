import { mount } from 'svelte';
import './app.css';
import App from './App.svelte';
import { applyTheme, settings } from './lib/settings.svelte';

// Set the theme attribute before the first paint so returning visitors never
// see a flash of the default tarot theme (app.css defaults to refined-tarot).
applyTheme(settings.theme);

const app = mount(App, {
  target: document.getElementById('app')!,
});

export default app;
