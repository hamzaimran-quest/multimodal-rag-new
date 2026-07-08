import type { AttachedImage, QuerySource } from "../types";

const DEFAULT_CAP = 2;
const DEDUP_IOU = 0.6;

function iou(a: number[], b: number[]): number {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = ix * iy;
  if (inter <= 0) return 0;
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  const union = areaA + areaB - inter;
  return union > 0 ? inter / union : 0;
}

function isDuplicate(candidate: AttachedImage, selected: AttachedImage[]): boolean {
  for (const existing of selected) {
    if (existing.image_chunk_id === candidate.image_chunk_id) return true;
    const samePage =
      existing.doc_id === candidate.doc_id && existing.page_number === candidate.page_number;
    if (samePage && existing.bbox && candidate.bbox) {
      if (iou(existing.bbox, candidate.bbox) > DEDUP_IOU) return true;
    }
  }
  return false;
}

/**
 * Derive the deduped, capped hero image strip from the persisted sources payload.
 * Mirrors the backend merge (intent images win, proximity fills), so it works
 * identically for live streaming and reloaded chat history without a separate
 * persisted field. Only explicitly promoted images are eligible — incidental
 * top-k image chunks stay in the Sources list, not the hero strip.
 */
export function deriveHeroImages(sources: QuerySource[], cap = DEFAULT_CAP): AttachedImage[] {
  const intent: AttachedImage[] = [];
  const proximity: AttachedImage[] = [];

  for (const source of sources) {
    if (source.attach_reason === "intent" && source.chunk_type === "image" && source.image_url) {
      intent.push({
        image_chunk_id: source.chunk_id,
        doc_id: source.doc_id ?? null,
        filename: source.filename,
        page_number: source.page_number,
        image_url: source.image_url,
        bbox: source.bbox ?? null,
        caption: source.snippet,
        score: source.score,
        reason: "intent",
      });
    }
    for (const attached of source.attached_images ?? []) {
      proximity.push(attached);
    }
  }

  const selected: AttachedImage[] = [];
  for (const image of intent.sort((a, b) => b.score - a.score)) {
    if (selected.length >= cap) break;
    if (!isDuplicate(image, selected)) selected.push(image);
  }
  for (const image of proximity.sort((a, b) => b.score - a.score)) {
    if (selected.length >= cap) break;
    if (!isDuplicate(image, selected)) selected.push(image);
  }
  return selected;
}
