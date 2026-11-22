/** Roman numerals for move numbers and arcana ordinals. */
export function toRoman(n: number): string {
  const table: Array<[number, string]> = [
    [1000, 'M'],
    [900, 'CM'],
    [500, 'D'],
    [400, 'CD'],
    [100, 'C'],
    [90, 'XC'],
    [50, 'L'],
    [40, 'XL'],
    [10, 'X'],
    [9, 'IX'],
    [5, 'V'],
    [4, 'IV'],
    [1, 'I'],
  ];
  let out = '';
  let v = n;
  for (const [val, sym] of table) {
    while (v >= val) {
      out += sym;
      v -= val;
    }
  }
  return out || '0';
}

/** Arcana identities used across the UI. */
export const ARCANA = {
  fool: { numeral: '0', name: 'The Fool' },
  priestess: { numeral: 'II', name: 'The Priestess' },
  emperor: { numeral: 'IV', name: 'The Emperor' },
  wheel: { numeral: 'X', name: 'The Wheel' },
} as const;
