import { useEffect, useState } from "react";

import { authFetch } from "../api/http";

interface AuthImageProps {
  src: string;
  alt: string;
  className?: string;
}

/** Renders an image from a protected API path via an authenticated blob URL. */
export function AuthImage({ src, alt, className }: AuthImageProps) {
  const [blobUrl, setBlobUrl] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    let objectUrl: string | null = null;

    authFetch(src)
      .then((response) => {
        if (!response.ok) throw new Error(`Image fetch failed: ${response.status}`);
        return response.blob();
      })
      .then((blob) => {
        if (!active) return;
        objectUrl = URL.createObjectURL(blob);
        setBlobUrl(objectUrl);
      })
      .catch(() => {
        if (active) setBlobUrl(null);
      });

    return () => {
      active = false;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [src]);

  if (!blobUrl) return null;
  return <img src={blobUrl} alt={alt} className={className} />;
}

/** Opens a protected image in a new tab using a short-lived blob URL. */
export async function openAuthenticatedImage(src: string): Promise<void> {
  const response = await authFetch(src);
  if (!response.ok) throw new Error(`Image fetch failed: ${response.status}`);
  const blob = await response.blob();
  const objectUrl = URL.createObjectURL(blob);
  window.open(objectUrl, "_blank", "noopener,noreferrer");
  window.setTimeout(() => URL.revokeObjectURL(objectUrl), 60_000);
}
