/// <reference types="svelte" />
/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL of the Fool's Gambit API (default: http://127.0.0.1:8000). */
  readonly VITE_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
