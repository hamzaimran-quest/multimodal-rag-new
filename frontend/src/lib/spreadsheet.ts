/** Excel-style column label: 1 -> A, 26 -> Z, 27 -> AA. */
export function columnLabel(index: number): string {
  let remaining = index;
  let label = "";
  while (remaining > 0) {
    remaining -= 1;
    label = String.fromCharCode(65 + (remaining % 26)) + label;
    remaining = Math.floor(remaining / 26);
  }
  return label;
}
