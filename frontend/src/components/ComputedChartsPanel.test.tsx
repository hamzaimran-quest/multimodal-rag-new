import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import type { ComputedChart } from "../types";
import { ComputedChartsPanel, computeYScale, placeBarLabelYs } from "./ComputedChartsPanel";

function makeBarChart(series: ComputedChart["series"]): ComputedChart {
  return {
    chunk_id: "chunk-1",
    filename: "huawei.pdf",
    page_number: 23,
    doc_id: "doc-1",
    chart_type: "bar",
    periods: ["2024", "2025"],
    series,
    value_axis_label: "CNY Million",
    period_axis_label: "Period",
    is_secondary: false,
    derivation: "computed",
    citation: {
      chunk_id: "chunk-1",
      filename: "huawei.pdf",
      page_number: 23,
      chunk_type: "table",
    },
  };
}

describe("computeYScale", () => {
  it("uses one yMax for ticks and bar math so the tallest datum fits", () => {
    const dataMax = 880_941;
    const { yMax, yTicks } = computeYScale(dataMax);

    expect(yMax).toBeGreaterThanOrEqual(dataMax);
    expect(yTicks[yTicks.length - 1]).toBe(yMax);
    expect(yMax).toBeGreaterThanOrEqual(dataMax * 1.08);
  });
});

describe("placeBarLabelYs", () => {
  it("staggers labels vertically when neighbor boxes would overlap", () => {
    const barTopY = 120;
    const labelYs = placeBarLabelYs([
      { key: "a", periodIdx: 0, centerX: 100, barTopY, labelText: "50,000" },
      { key: "b", periodIdx: 0, centerX: 120, barTopY, labelText: "52,000" },
    ]);

    expect(labelYs.get("a")).toBe(barTopY - 4);
    expect(labelYs.get("b")).not.toBe(barTopY - 4);
    expect(labelYs.get("b")!).toBeLessThan(labelYs.get("a")!);
  });
});

describe("ComputedChartsPanel", () => {
  it("renders exact value labels on every bar including short values", () => {
    const chart = makeBarChart([
      { name: "China", values: [880_941, 900_000] },
      { name: "EMEA", values: [180_000, 190_000] },
      { name: "Other", values: [42_500, 44_100] },
      { name: "Total", values: [880_941, 900_000] },
    ]);

    render(
      <ComputedChartsPanel charts={[chart]} isOpen onToggleOpen={() => undefined} messageIndex={0} />,
    );

    expect(screen.getAllByText("880,941").length).toBeGreaterThan(0);
    expect(screen.getAllByText("900,000").length).toBeGreaterThan(0);
    expect(screen.getAllByTestId("bar-label-Other-0")).toHaveLength(1);
    expect(screen.getByTestId("bar-label-Other-0")).toHaveTextContent("42,500");
  });

  it("shows precise tooltip on hover", async () => {
    const user = userEvent.setup();
    const chart = makeBarChart([
      { name: "China", values: [880_941, 900_000] },
      { name: "Other", values: [42_500, 44_100] },
    ]);

    render(
      <ComputedChartsPanel charts={[chart]} isOpen onToggleOpen={() => undefined} messageIndex={0} />,
    );

    const shortBar = screen.getByLabelText("Other, 2024, 42,500");
    await user.hover(shortBar);

    const tooltip = await screen.findByTestId("chart-tooltip");
    expect(tooltip).toHaveTextContent("Other · 2024");
    expect(tooltip).toHaveTextContent("42,500 CNY Million");
  });
});
