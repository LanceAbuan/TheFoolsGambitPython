/**
 * The Fool's mark: a knight silhouette drawn as geometry rather than a font
 * glyph. The `♞` character rendered as a random system-symbol blob at small
 * sizes (different on every platform), so the brand is an explicit path.
 */
export function Knight({ size = 22 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden
      focusable="false"
      style={{ display: 'block', flex: '0 0 auto' }}
    >
      <path
        d="M8.4 20.2h8.3c.6 0 1-.4 1-1v-1.1c0-.5-.4-.9-.9-.9h-.7l-1.5-4.4c3.4-1.4 5.2-4.1 5.2-7.3V4.3c0-.8-1-1.1-1.4-.5l-1.9 2.6c-.2.3-.6.4-.9.3L13.6 6c-.4-.1-.8.1-1 .5L12 8.3c-.1.3-.4.5-.7.5H8.9c-.8 0-1.5.6-1.5 1.4 0 .5.3 1 .7 1.2l1.6.9c.3.2.4.5.3.8l-.6 1.6c-.1.3-.4.5-.7.5H7.1c-.5 0-.9.4-.9.9v1.6c0 .4.2.7.6.9l1.3.6Z"
        fill="currentColor"
      />
      <path d="M7.6 21.6h9.2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}
