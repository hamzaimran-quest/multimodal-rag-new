import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { MarkdownAnswer } from "./MarkdownAnswer";

describe("MarkdownAnswer", () => {
  it("renders bullet lists and headings", () => {
    render(
      <MarkdownAnswer
        content={`## Highlights\n\n- First item\n- Second item`}
      />,
    );

    expect(screen.getByRole("heading", { level: 2, name: "Highlights" })).toBeInTheDocument();
    expect(screen.getByText("First item")).toBeInTheDocument();
    expect(screen.getByText("Second item")).toBeInTheDocument();
  });

  it("renders markdown tables", () => {
    render(
      <MarkdownAnswer
        content={`| Metric | 2024 | 2025 |
| --- | --- | --- |
| Revenue | 100 | 120 |`}
      />,
    );

    expect(screen.getByRole("columnheader", { name: "Metric" })).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "120" })).toBeInTheDocument();
  });

  it("renders a table glued to a list item without a blank line", () => {
    render(
      <MarkdownAnswer
        content={`Some examples of lists:
* A table on page 17, showing items and their quantities:
| ITEM | NEEDED |
| --- | --- |
| Books | 1 |
| Pens | 3 |`}
      />,
    );

    expect(screen.getByRole("columnheader", { name: "ITEM" })).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "Books" })).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "3" })).toBeInTheDocument();
  });
});
