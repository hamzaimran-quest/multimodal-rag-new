export function formatUploadDate(iso: string | null): string {
  if (!iso) return "—";
  try {
    return new Intl.DateTimeFormat(undefined, {
      dateStyle: "medium",
      timeStyle: "short",
    }).format(new Date(iso));
  } catch {
    return iso;
  }
}

export function statusLabel(status: string): string {
  switch (status) {
    case "indexed":
      return "Indexed";
    case "processing":
      return "Processing";
    case "pending":
      return "Pending";
    case "failed":
      return "Failed";
    default:
      return status;
  }
}

export function statusClassName(status: string): string {
  switch (status) {
    case "indexed":
      return "bg-emerald-500/15 text-emerald-300 ring-emerald-500/30";
    case "processing":
    case "pending":
      return "bg-amber-500/15 text-amber-200 ring-amber-500/30";
    case "failed":
      return "bg-rose-500/15 text-rose-300 ring-rose-500/30";
    default:
      return "bg-slate-500/15 text-slate-300 ring-slate-500/30";
  }
}

export function estimateIngestionProgress(
  status: string,
  uploadedAtIso: string | null,
): number {
  if (status === "indexed") return 100;
  if (status === "failed") return 100;

  const uploadedAt = uploadedAtIso ? Date.parse(uploadedAtIso) : NaN;
  const elapsedMs = Number.isNaN(uploadedAt) ? 0 : Math.max(Date.now() - uploadedAt, 0);
  const elapsedSeconds = elapsedMs / 1000;

  if (status === "pending") {
    return Math.min(20, 5 + elapsedSeconds * 1.2);
  }
  if (status === "processing") {
    return Math.min(95, 20 + elapsedSeconds * 2.3);
  }
  return 0;
}
