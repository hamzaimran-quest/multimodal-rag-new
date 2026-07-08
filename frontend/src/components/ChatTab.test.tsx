import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { ChatTab } from "./ChatTab";

vi.mock("../api/query", () => ({
  streamQuery: vi.fn(async (_payload, handlers) => {
    handlers.onToken("Revenue ");
    handlers.onToken("increased.");
    handlers.onSources([
      {
        chunk_id: "c1",
        filename: "huawei.pdf",
        page_number: 12,
        chunk_type: "text",
        snippet: "Five-Year Financial Highlights",
        score: 0.9,
      },
    ]);
  }),
}));

describe("ChatTab", () => {
  it("streams answer and renders sources", async () => {
    render(<ChatTab />);
    const user = userEvent.setup();

    await user.type(screen.getByTestId("chat-input"), "summarize highlights");
    await user.click(screen.getByTestId("chat-send"));

    expect(await screen.findByText("Revenue increased.")).toBeInTheDocument();
    expect(await screen.findByText(/huawei\.pdf · page 12 · text/i)).toBeInTheDocument();
    expect(await screen.findByText("Five-Year Financial Highlights")).toBeInTheDocument();
  });
});

