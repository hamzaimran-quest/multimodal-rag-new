export interface QuerySource {
  chunk_id: string;
  filename: string;
  page_number: number;
  chunk_type: "text" | "table" | "image" | string;
  snippet: string;
  image_url?: string | null;
  score: number;
}

export interface ChartSeries {
  name: string;
  values: number[];
}

export interface ComputedChart {
  chunk_id: string;
  filename: string;
  page_number: number;
  doc_id: string;
  chart_type: "bar" | "line";
  chart_url?: string;
  title?: string | null;
  periods?: string[];
  series?: ChartSeries[];
  value_axis_label?: string;
  period_axis_label?: string;
  derivation: "tool";
  citation: {
    chunk_id: string;
    filename: string;
    page_number: number;
    chunk_type: string;
  };
}

/** Chart with series/periods guaranteed for legacy SVG rendering. */
export type SvgComputedChart = ComputedChart & {
  periods: string[];
  series: ChartSeries[];
};

export function normalizeChartForSvg(chart: ComputedChart): SvgComputedChart {
  return {
    ...chart,
    periods: chart.periods ?? [],
    series: chart.series ?? [],
  };
}
