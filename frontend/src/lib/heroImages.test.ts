import { describe, expect, it } from "vitest";

import type { QuerySource } from "../types";
import { deriveHeroImages } from "./heroImages";

function textSource(overrides: Partial<QuerySource> = {}): QuerySource {
  return {
    chunk_id: "t1",
    doc_id: "d1",
    filename: "report.pdf",
    page_number: 3,
    chunk_type: "text",
    snippet: "Chairman of the Board",
    score: 1.0,
    source_format: "pdf",
    ...overrides,
  };
}

describe("deriveHeroImages", () => {
  it("promotes proximity-attached images", () => {
    const sources: QuerySource[] = [
      textSource({
        attached_images: [
          {
            image_chunk_id: "img1",
            doc_id: "d1",
            filename: "report.pdf",
            page_number: 3,
            image_url: "/images/d1/img1.png",
            bbox: [0, 200, 100, 320],
            score: 0.8,
            reason: "proximity",
          },
        ],
      }),
    ];
    const hero = deriveHeroImages(sources);
    expect(hero.map((h) => h.image_chunk_id)).toEqual(["img1"]);
  });

  it("ignores incidental image sources without an attach reason", () => {
    const sources: QuerySource[] = [
      {
        chunk_id: "img9",
        doc_id: "d1",
        filename: "report.pdf",
        page_number: 5,
        chunk_type: "image",
        snippet: "",
        image_url: "/images/d1/img9.png",
        score: 0.4,
        source_format: "pdf",
      },
    ];
    expect(deriveHeroImages(sources)).toEqual([]);
  });

  it("prioritizes intent images and caps the result", () => {
    const sources: QuerySource[] = [
      {
        chunk_id: "intent1",
        doc_id: "d1",
        filename: "report.pdf",
        page_number: 2,
        chunk_type: "image",
        snippet: "",
        image_url: "/images/d1/intent1.png",
        score: 0.9,
        source_format: "pdf",
        attach_reason: "intent",
        bbox: [0, 0, 50, 50],
      },
      textSource({
        attached_images: [
          {
            image_chunk_id: "prox1",
            doc_id: "d1",
            page_number: 3,
            image_url: "/images/d1/prox1.png",
            bbox: [300, 300, 360, 360],
            score: 0.95,
            reason: "proximity",
          },
        ],
      }),
    ];
    const hero = deriveHeroImages(sources, 1);
    expect(hero.map((h) => h.image_chunk_id)).toEqual(["intent1"]);
  });

  it("caps explicit visual-intent images at one by default", () => {
    const intentImage = (id: string, score: number): QuerySource => ({
      chunk_id: id,
      doc_id: "d1",
      filename: "report.pdf",
      page_number: 2,
      chunk_type: "image",
      snippet: "",
      image_url: `/images/d1/${id}.png`,
      score,
      source_format: "pdf",
      attach_reason: "intent",
      bbox: [0, 0, 50, 50],
    });
    const sources: QuerySource[] = [
      intentImage("intent1", 0.9),
      intentImage("intent2", 0.85),
    ];
    const hero = deriveHeroImages(sources);
    expect(hero.map((h) => h.image_chunk_id)).toEqual(["intent1"]);
  });
});
