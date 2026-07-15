import { useState } from "react";

import type { AttachedImage } from "../types";
import { AuthImage } from "./AuthImage";

interface HeroImagesProps {
  images: AttachedImage[];
}

function locationLabel(image: AttachedImage): string {
  const name = image.filename ?? "document";
  if (image.page_number) return `${name} · page ${image.page_number}`;
  return name;
}

/** Prominent image strip shown above the Sources panel when the query is
 * explicitly visual or an image sits next to the top text evidence. Each
 * thumbnail opens an overlay for a clearer, larger view. */
export function HeroImages({ images }: HeroImagesProps) {
  const [openImageId, setOpenImageId] = useState<string | null>(null);

  if (images.length === 0) return null;

  const openImage = images.find((image) => image.image_chunk_id === openImageId) ?? null;

  return (
    <div className="mt-3 flex flex-wrap gap-3" data-testid="hero-images">
      {images.map((image) => (
        <figure
          key={image.image_chunk_id}
          className="overflow-hidden rounded-[10px] border border-[#2a2a2a] bg-[#141414]"
        >
          <button
            type="button"
            onClick={() => setOpenImageId(image.image_chunk_id)}
            className="block cursor-pointer"
            aria-label={`Open larger view of image from ${locationLabel(image)}`}
            data-testid={`open-hero-image-${image.image_chunk_id}`}
          >
            <AuthImage
              src={image.image_url}
              alt={image.caption || locationLabel(image)}
              className="max-h-64 w-auto max-w-[320px] object-contain"
            />
          </button>
          <figcaption className="border-t border-[#2a2a2a] px-2.5 py-1.5 text-[11px] text-[#737373]">
            {locationLabel(image)}
          </figcaption>
        </figure>
      ))}

      {openImage && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
          data-testid="hero-image-overlay"
          role="dialog"
          aria-modal="true"
          aria-label="Image viewer"
        >
          <button
            type="button"
            aria-label="Close image"
            className="absolute inset-0 cursor-default"
            onClick={() => setOpenImageId(null)}
          />
          <div
            className="relative z-10 flex max-h-[90vh] w-full max-w-3xl flex-col overflow-hidden rounded-[16px] border border-[#2a2a2a] bg-gradient-to-b from-[#1a1a1a] to-[#111111] shadow-2xl"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-[#2a2a2a] px-4 py-3">
              <span className="text-[13px] font-medium text-[#e5e5e5]">{locationLabel(openImage)}</span>
              <button
                type="button"
                onClick={() => setOpenImageId(null)}
                className="rounded-[6px] border border-[#333333] px-2.5 py-1 text-[11px] font-medium text-[#d4d4d4] transition-colors hover:border-[#525252] hover:text-[#f5f5f5]"
                data-testid="close-hero-image"
              >
                Close
              </button>
            </div>
            <div className="flex justify-center overflow-auto p-3" data-testid="hero-image-panel">
              <AuthImage
                src={openImage.image_url}
                alt={openImage.caption || locationLabel(openImage)}
                className="max-h-[75vh] w-auto max-w-full object-contain"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
