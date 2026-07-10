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

export function cellHighlightClass(
  rowNumber: number,
  colNumber: number,
  rowRange?: number[] | null,
  colRange?: number[] | null,
): string {
  if (!rowRange || rowRange.length !== 2 || !colRange || colRange.length !== 2) {
    return "";
  }
  const [rowStart, rowEnd] = rowRange;
  const [colStart, colEnd] = colRange;
  if (rowNumber >= rowStart && rowNumber <= rowEnd && colNumber >= colStart && colNumber <= colEnd) {
    return "bg-amber-500/20 ring-1 ring-inset ring-amber-400/60";
  }
  return "";
}
