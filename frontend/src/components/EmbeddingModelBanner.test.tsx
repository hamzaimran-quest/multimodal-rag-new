import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import * as client from "../api/client";
import { EmbeddingModelBanner } from "./EmbeddingModelBanner";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("EmbeddingModelBanner", () => {
  it("renders nothing once the model is already ready", async () => {
    vi.spyOn(client, "getEmbeddingModelStatus").mockResolvedValue({
      state: "ready",
      started_at: 100,
      finished_at: 102,
      error: null,
    });

    render(<EmbeddingModelBanner />);

    await waitFor(() => {
      expect(screen.queryByTestId("embedding-model-banner")).not.toBeInTheDocument();
    });
  });

  it("shows a loading message while the model is warming up", async () => {
    vi.spyOn(client, "getEmbeddingModelStatus").mockResolvedValue({
      state: "loading",
      started_at: Date.now() / 1000,
      finished_at: null,
      error: null,
    });

    render(<EmbeddingModelBanner />);

    const banner = await screen.findByTestId("embedding-model-banner");
    expect(banner).toHaveTextContent("Loading embedding model");
  });

  it("disappears once status flips to ready on a later poll", async () => {
    let call = 0;
    vi.spyOn(client, "getEmbeddingModelStatus").mockImplementation(async () => {
      call += 1;
      if (call === 1) {
        return { state: "loading", started_at: Date.now() / 1000, finished_at: null, error: null };
      }
      return { state: "ready", started_at: 100, finished_at: 105, error: null };
    });

    render(<EmbeddingModelBanner />);

    await screen.findByTestId("embedding-model-banner");
    await waitFor(
      () => {
        expect(screen.queryByTestId("embedding-model-banner")).not.toBeInTheDocument();
      },
      { timeout: 3000 },
    );
  });
});
