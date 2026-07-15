import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { AttachedImage } from "../types";
import { HeroImages } from "./HeroImages";

vi.mock("./AuthImage", () => ({
  AuthImage: ({ alt }: { alt: string }) => <img alt={alt} data-testid="auth-image-stub" />,
}));

function makeImage(overrides: Partial<AttachedImage> = {}): AttachedImage {
  return {
    image_chunk_id: "img-1",
    doc_id: "doc-1",
    filename: "NASDAQ_AAPL_2025_removed.pdf",
    page_number: 23,
    image_url: "/images/doc-1/img-1.png",
    caption: "Company stock performance chart",
    score: 0.9,
    reason: "intent",
    ...overrides,
  };
}

describe("HeroImages", () => {
  it("renders nothing for an empty image list", () => {
    const { container } = render(<HeroImages images={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it("opens a clicked thumbnail in an overlay with a close control", async () => {
    const user = userEvent.setup();
    render(<HeroImages images={[makeImage()]} />);

    expect(screen.queryByTestId("hero-image-overlay")).not.toBeInTheDocument();

    await user.click(screen.getByTestId("open-hero-image-img-1"));

    expect(screen.getByTestId("hero-image-overlay")).toBeInTheDocument();
    expect(screen.getByTestId("close-hero-image")).toBeInTheDocument();
    expect(screen.getAllByText("NASDAQ_AAPL_2025_removed.pdf · page 23").length).toBeGreaterThan(0);
  });

  it("closes the overlay via the close button", async () => {
    const user = userEvent.setup();
    render(<HeroImages images={[makeImage()]} />);

    await user.click(screen.getByTestId("open-hero-image-img-1"));
    expect(screen.getByTestId("hero-image-overlay")).toBeInTheDocument();

    await user.click(screen.getByTestId("close-hero-image"));
    expect(screen.queryByTestId("hero-image-overlay")).not.toBeInTheDocument();
  });

  it("closes the overlay via the backdrop", async () => {
    const user = userEvent.setup();
    render(<HeroImages images={[makeImage()]} />);

    await user.click(screen.getByTestId("open-hero-image-img-1"));
    await user.click(screen.getByLabelText("Close image"));

    expect(screen.queryByTestId("hero-image-overlay")).not.toBeInTheDocument();
  });
});
