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
  periods: string[];
  series: ChartSeries[];
  value_axis_label?: string;
  period_axis_label?: string;
  is_secondary: boolean;
  derivation: "computed";
  citation: {
    chunk_id: string;
    filename: string;
    page_number: number;
    chunk_type: string;
  };
}
