import type { AttachedImage } from "../types";
import { AuthImage } from "./AuthImage";

interface HeroImagesProps {
  images: AttachedImage[];
}

function locationLabel(image: AttachedImage): string {
  const name = image.filename ?? "document";
  if (image.source_format === "docx") {
    if (image.section) return `${name} · ${image.section}`;
    if (image.page_number && image.page_number > 0) return name;
    return name;
  }
  if (image.page_number) return `${name} · page ${image.page_number}`;
  return name;
}

/** Prominent image strip shown above the Sources panel when the query is
 * explicitly visual or an image sits next to the top text evidence. */
export function HeroImages({ images }: HeroImagesProps) {
  if (images.length === 0) return null;

  return (
    <div className="mt-3 flex flex-wrap gap-3" data-testid="hero-images">
      {images.map((image) => (
        <figure
          key={image.image_chunk_id}
          className="overflow-hidden rounded-[10px] border border-[#2a2a2a] bg-[#141414]"
        >
          <AuthImage
            src={image.image_url}
            alt={image.caption || locationLabel(image)}
            className="max-h-64 w-auto max-w-[320px] object-contain"
          />
          <figcaption className="border-t border-[#2a2a2a] px-2.5 py-1.5 text-[11px] text-[#737373]">
            {locationLabel(image)}
          </figcaption>
        </figure>
      ))}
    </div>
  );
}
