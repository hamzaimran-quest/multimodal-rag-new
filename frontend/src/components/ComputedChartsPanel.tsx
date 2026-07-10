import { useState } from "react";

import type { ChartSeries, ComputedChart, SvgComputedChart } from "../types";
import { normalizeChartForSvg } from "../types";

const CHART_COLORS = [
  "#60a5fa",
  "#34d399",
  "#fbbf24",
  "#f472b6",
  "#a78bfa",
  "#fb923c",
  "#22d3ee",
  "#f87171",
];

interface ComputedChartsPanelProps {
  charts: ComputedChart[];
  isOpen: boolean;
  onToggleOpen: () => void;
  messageIndex: number;
}

interface ChartLayout {
  width: number;
  height: number;
  padding: { top: number; right: number; bottom: number; left: number };
  innerW: number;
  innerH: number;
}

function chartLayout(): ChartLayout {
  const width = 520;
  const height = 260;
  const padding = { top: 20, right: 20, bottom: 62, left: 72 };
  return {
    width,
    height,
    padding,
    innerW: width - padding.left - padding.right,
    innerH: height - padding.top - padding.bottom,
  };
}

function formatValue(value: number): string {
  if (Math.abs(value) >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (Math.abs(value) >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  if (Number.isInteger(value)) return String(value);
  return value.toFixed(1);
}

function formatExactValue(value: number): string {
  if (!Number.isFinite(value)) return "—";
  const rounded = Math.abs(value - Math.round(value)) < 1e-9 ? Math.round(value) : value;
  if (Number.isInteger(rounded)) {
    return rounded.toLocaleString("en-US");
  }
  return rounded.toLocaleString("en-US", { maximumFractionDigits: 2 });
}

const BAR_LABEL_FONT_SIZE = 8;
const BAR_LABEL_HEIGHT = 10;
/** Approximate glyph width for 8px tabular labels in SVG user units. */
const BAR_LABEL_CHAR_WIDTH = 4.8;
const BAR_LABEL_COLLISION_PADDING = 1;
const BAR_LABEL_OFFSETS_ABOVE_BAR = [4, 8, 20, 32] as const;

interface LabelBox {
  left: number;
  right: number;
  top: number;
  bottom: number;
}

interface BarLabelInput {
  key: string;
  periodIdx: number;
  centerX: number;
  barTopY: number;
  labelText: string;
}

function estimateLabelWidth(text: string): number {
  return text.length * BAR_LABEL_CHAR_WIDTH;
}

function estimateLabelBox(centerX: number, barTopY: number, text: string, offsetAboveBar: number): LabelBox {
  const halfWidth = estimateLabelWidth(text) / 2;
  const labelY = barTopY - offsetAboveBar;
  return {
    left: centerX - halfWidth,
    right: centerX + halfWidth,
    top: labelY - BAR_LABEL_HEIGHT,
    bottom: labelY + 2,
  };
}

function labelBoxesOverlap(a: LabelBox, b: LabelBox): boolean {
  return !(
    a.right + BAR_LABEL_COLLISION_PADDING < b.left
    || b.right + BAR_LABEL_COLLISION_PADDING < a.left
    || a.bottom + BAR_LABEL_COLLISION_PADDING < b.top
    || b.bottom + BAR_LABEL_COLLISION_PADDING < a.top
  );
}

/** Assign label Y positions; stagger vertically only when neighbor boxes would overlap. */
export function placeBarLabelYs(bars: BarLabelInput[]): Map<string, number> {
  const labelYs = new Map<string, number>();
  const byPeriod = new Map<number, BarLabelInput[]>();

  for (const bar of bars) {
    const periodBars = byPeriod.get(bar.periodIdx) ?? [];
    periodBars.push(bar);
    byPeriod.set(bar.periodIdx, periodBars);
  }

  for (const periodBars of byPeriod.values()) {
    periodBars.sort((a, b) => a.centerX - b.centerX);
    const placed: LabelBox[] = [];

    for (const bar of periodBars) {
      let labelY = bar.barTopY - BAR_LABEL_OFFSETS_ABOVE_BAR[0];

      for (const offset of BAR_LABEL_OFFSETS_ABOVE_BAR) {
        const candidateBox = estimateLabelBox(bar.centerX, bar.barTopY, bar.labelText, offset);
        const collides = placed.some((box) => labelBoxesOverlap(candidateBox, box));
        labelY = bar.barTopY - offset;
        if (!collides) {
          placed.push(candidateBox);
          break;
        }
        if (offset === BAR_LABEL_OFFSETS_ABOVE_BAR[BAR_LABEL_OFFSETS_ABOVE_BAR.length - 1]) {
          placed.push(candidateBox);
        }
      }

      labelYs.set(bar.key, labelY);
    }
  }

  return labelYs;
}

interface ChartDatumHover {
  key: string;
  seriesName: string;
  period: string;
  value: number;
  valueAxisLabel: string;
  x: number;
  y: number;
  color: string;
}

function ChartTooltip({ hover, layout }: { hover: ChartDatumHover; layout: ChartLayout }) {
  const title = `${hover.seriesName} · ${hover.period}`;
  const valueText = `${formatExactValue(hover.value)}${hover.valueAxisLabel ? ` ${hover.valueAxisLabel}` : ""}`;

  return (
    <div
      className="pointer-events-none absolute z-10 max-w-[220px] -translate-x-1/2 -translate-y-[calc(100%+6px)] rounded-[6px] border border-[#404040] bg-[#262626] px-2.5 py-1.5 text-left shadow-lg shadow-black/40"
      style={{
        left: `${(hover.x / layout.width) * 100}%`,
        top: `${(hover.y / layout.height) * 100}%`,
      }}
      data-testid="chart-tooltip"
    >
      <p className="text-[10px] font-medium text-[#e5e5e5]">{title}</p>
      <p className="mt-0.5 text-[10px] tabular-nums text-[#d4d4d4]">{valueText}</p>
    </div>
  );
}

function niceStep(raw: number): number {
  if (raw <= 0) return 1;
  const exponent = Math.floor(Math.log10(raw));
  const fraction = raw / 10 ** exponent;
  let niceFraction = 10;
  if (fraction <= 1) niceFraction = 1;
  else if (fraction <= 2) niceFraction = 2;
  else if (fraction <= 5) niceFraction = 5;
  return niceFraction * 10 ** exponent;
}

/** Headroom above the tallest datum so bars and labels stay inside the plot. */
const Y_AXIS_HEADROOM = 1.08;

export interface YScale {
  yMax: number;
  yTicks: number[];
}

/**
 * Single source of truth for the value axis. Bar heights, point positions, and
 * tick/grid placement must all use the returned yMax — never derive a second scale.
 */
export function computeYScale(dataMax: number, tickCount = 4): YScale {
  if (dataMax <= 0) {
    return { yMax: 1, yTicks: [0, 1] };
  }

  const domainTarget = dataMax * Y_AXIS_HEADROOM;
  const step = niceStep(domainTarget / tickCount);
  const yTicks: number[] = [];

  for (let tick = 0; tick <= domainTarget; tick += step) {
    yTicks.push(tick);
  }

  let yMax = yTicks[yTicks.length - 1] ?? step;
  while (yMax < dataMax) {
    yMax += step;
    yTicks.push(yMax);
  }

  if (yTicks.length < 2) {
    const fallback = Math.max(dataMax, 1);
    return { yMax: fallback, yTicks: [0, fallback] };
  }

  return { yMax, yTicks };
}

function valueToBarTop(value: number, yMax: number, layout: ChartLayout): number {
  const barHeight = (value / yMax) * layout.innerH;
  return layout.padding.top + layout.innerH - barHeight;
}

function clampTooltipAnchorY(barTopY: number, plotTop: number): number {
  return Math.max(plotTop, barTopY);
}

function ChartAxes({
  layout,
  yTicks,
  yMax,
  periods,
  periodAxisLabel,
  valueAxisLabel,
}: {
  layout: ChartLayout;
  yTicks: number[];
  yMax: number;
  periods: string[];
  periodAxisLabel: string;
  valueAxisLabel: string;
}) {
  const { height, padding, innerW, innerH } = layout;

  return (
    <>
      <line
        x1={padding.left}
        y1={padding.top}
        x2={padding.left}
        y2={padding.top + innerH}
        stroke="#404040"
        strokeWidth="1"
      />
      <line
        x1={padding.left}
        y1={padding.top + innerH}
        x2={padding.left + innerW}
        y2={padding.top + innerH}
        stroke="#404040"
        strokeWidth="1"
      />

      {yTicks.map((tick) => {
        const y = padding.top + innerH - (tick / yMax) * innerH;
        return (
          <g key={`y-${tick}`}>
            <line x1={padding.left - 4} y1={y} x2={padding.left + innerW} y2={y} stroke="#262626" strokeWidth="1" />
            <text x={padding.left - 8} y={y + 3} textAnchor="end" className="fill-[#737373] text-[9px]">
              {formatValue(tick)}
            </text>
          </g>
        );
      })}

      {periods.map((period: string, periodIdx: number) => {
        const x = padding.left + (periodIdx + 0.5) * (innerW / Math.max(periods.length, 1));
        return (
          <text key={period} x={x} y={height - 30} textAnchor="middle" className="fill-[#a3a3a3] text-[10px]">
            {period}
          </text>
        );
      })}

      <text x={padding.left + innerW / 2} y={height - 10} textAnchor="middle" className="fill-[#737373] text-[10px]">
        {periodAxisLabel}
      </text>

      <text
        x={16}
        y={padding.top + innerH / 2}
        textAnchor="middle"
        transform={`rotate(-90 16 ${padding.top + innerH / 2})`}
        className="fill-[#737373] text-[10px]"
      >
        {valueAxisLabel}
      </text>
    </>
  );
}

function barChartLayout(): ChartLayout {
  const width = 520;
  const height = 280;
  const padding = { top: 36, right: 20, bottom: 62, left: 72 };
  return {
    width,
    height,
    padding,
    innerW: width - padding.left - padding.right,
    innerH: height - padding.top - padding.bottom,
  };
}

interface BarRenderDatum {
  key: string;
  seriesName: string;
  period: string;
  periodIdx: number;
  value: number;
  x: number;
  barTopY: number;
  barHeight: number;
  centerX: number;
  color: string;
  labelText: string;
}

function buildBarRenderData(
  chart: SvgComputedChart,
  layout: ChartLayout,
  yMax: number,
  barWidth: number,
  groupWidth: number,
): BarRenderDatum[] {
  const { padding, innerH } = layout;
  const seriesCount = chart.series.length;
  const bars: BarRenderDatum[] = [];

  chart.series.forEach((series: ChartSeries, seriesIdx: number) => {
    series.values.forEach((value: number, periodIdx: number) => {
      const barHeight = (value / yMax) * innerH;
      const groupX = padding.left + periodIdx * groupWidth + groupWidth / 2;
      const offset = (seriesIdx - (seriesCount - 1) / 2) * (barWidth + 2);
      const x = groupX + offset - barWidth / 2;
      const barTopY = valueToBarTop(value, yMax, layout);
      const key = `${series.name}-${periodIdx}`;

      bars.push({
        key,
        seriesName: series.name,
        period: chart.periods[periodIdx] ?? String(periodIdx + 1),
        periodIdx,
        value,
        x,
        barTopY,
        barHeight,
        centerX: x + barWidth / 2,
        color: CHART_COLORS[seriesIdx % CHART_COLORS.length],
        labelText: formatExactValue(value),
      });
    });
  });

  return bars;
}

function BarChartSvg({ chart }: { chart: SvgComputedChart }) {
  const layout = barChartLayout();
  const { width, height, padding, innerW, innerH } = layout;
  const [hovered, setHovered] = useState<ChartDatumHover | null>(null);
  const dataMax = Math.max(...chart.series.flatMap((s: ChartSeries) => s.values), 0);
  const { yMax, yTicks } = computeYScale(dataMax);
  const groupCount = chart.periods.length;
  const seriesCount = chart.series.length;
  const groupWidth = innerW / Math.max(groupCount, 1);
  const barWidth = Math.min(18, (groupWidth - 8) / Math.max(seriesCount, 1));
  const valueAxisLabel = chart.value_axis_label ?? "Value";
  const periodAxisLabel = chart.period_axis_label ?? "Period";
  const bars = buildBarRenderData(chart, layout, yMax, barWidth, groupWidth);
  const labelYs = placeBarLabelYs(
    bars.map((bar) => ({
      key: bar.key,
      periodIdx: bar.periodIdx,
      centerX: bar.centerX,
      barTopY: bar.barTopY,
      labelText: bar.labelText,
    })),
  );

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${width} ${height}`} className="h-auto w-full" role="img" aria-label="Computed bar chart">
        <ChartAxes
          layout={layout}
          yTicks={yTicks}
          yMax={yMax}
          periods={chart.periods}
          periodAxisLabel={periodAxisLabel}
          valueAxisLabel={valueAxisLabel}
        />
        {bars.map((bar) => {
          const hoverPayload: ChartDatumHover = {
            key: bar.key,
            seriesName: bar.seriesName,
            period: bar.period,
            value: bar.value,
            valueAxisLabel,
            x: bar.centerX,
            y: clampTooltipAnchorY(bar.barTopY, padding.top),
            color: bar.color,
          };

          return (
            <g key={bar.key}>
              <rect
                x={bar.x}
                y={padding.top}
                width={barWidth}
                height={innerH}
                fill="transparent"
                className="cursor-pointer"
                onMouseEnter={() => setHovered(hoverPayload)}
                onMouseLeave={() => setHovered((current) => (current?.key === bar.key ? null : current))}
                aria-label={`${bar.seriesName}, ${bar.period}, ${bar.labelText}`}
              />
              <rect
                x={bar.x}
                y={bar.barTopY}
                width={barWidth}
                height={Math.max(bar.barHeight, 0)}
                fill={bar.color}
                rx={2}
                pointerEvents="none"
              />
              <text
                x={bar.centerX}
                y={labelYs.get(bar.key) ?? bar.barTopY - BAR_LABEL_OFFSETS_ABOVE_BAR[0]}
                textAnchor="middle"
                className="fill-[#e5e5e5] pointer-events-none select-none"
                fontSize={BAR_LABEL_FONT_SIZE}
                data-testid={`bar-label-${bar.key}`}
              >
                {bar.labelText}
              </text>
            </g>
          );
        })}
      </svg>
      {hovered && <ChartTooltip hover={hovered} layout={layout} />}
    </div>
  );
}

function LineChartSvg({ chart }: { chart: SvgComputedChart }) {
  const layout = chartLayout();
  const { padding, innerW, width, height } = layout;
  const [hovered, setHovered] = useState<ChartDatumHover | null>(null);
  const series = chart.series[0];
  const dataMax = Math.max(...series.values, 0);
  const { yMax, yTicks } = computeYScale(dataMax);
  const valueAxisLabel = chart.value_axis_label ?? "Value";
  const periodAxisLabel = chart.period_axis_label ?? "Period";
  const points = series.values.map((value: number, idx: number) => {
    const x = padding.left + (idx + 0.5) * (innerW / Math.max(series.values.length, 1));
    const pointY = valueToBarTop(value, yMax, layout);
    return { x, y: pointY, value, period: chart.periods[idx] ?? String(idx + 1) };
  });

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${width} ${height}`} className="h-auto w-full" role="img" aria-label="Computed line chart">
        <ChartAxes
          layout={layout}
          yTicks={yTicks}
          yMax={yMax}
          periods={chart.periods}
          periodAxisLabel={periodAxisLabel}
          valueAxisLabel={valueAxisLabel}
        />
        <polyline
          fill="none"
          stroke={CHART_COLORS[0]}
          strokeWidth="2.5"
          points={points.map((point) => `${point.x},${point.y}`).join(" ")}
        />
        {points.map((point, idx: number) => {
          const key = periodLabel(chart.periods[idx], idx);
          const hoverPayload: ChartDatumHover = {
            key,
            seriesName: series.name,
            period: point.period,
            value: point.value,
            valueAxisLabel,
            x: point.x,
            y: clampTooltipAnchorY(point.y, padding.top),
            color: CHART_COLORS[0],
          };

          return (
            <g key={key}>
              <circle
                cx={point.x}
                cy={point.y}
                r={10}
                fill="transparent"
                className="cursor-pointer"
                onMouseEnter={() => setHovered(hoverPayload)}
                onMouseLeave={() => setHovered((current) => (current?.key === key ? null : current))}
                aria-label={`${series.name}, ${point.period}, ${formatExactValue(point.value)}`}
              />
              <circle
                cx={point.x}
                cy={point.y}
                r={4}
                fill={CHART_COLORS[0]}
                stroke="#171717"
                strokeWidth="1.5"
                pointerEvents="none"
              />
              <text
                x={point.x}
                y={point.y - 8}
                textAnchor="middle"
                className="fill-[#e5e5e5] text-[8px] pointer-events-none select-none"
                data-testid={`line-label-${key}`}
              >
                {formatExactValue(point.value)}
              </text>
            </g>
          );
        })}
      </svg>
      {hovered && <ChartTooltip hover={hovered} layout={layout} />}
    </div>
  );
}

function periodLabel(label: string, idx: number): string {
  return `${label}-${idx}`;
}

function ChartLegend({ chart }: { chart: SvgComputedChart }) {
  if (chart.chart_type === "line" && chart.series.length === 1) {
    return (
      <p className="mt-2 text-[11px] text-[#737373]">
        <span className="inline-block h-2 w-2 rounded-full mr-1.5" style={{ background: CHART_COLORS[0] }} />
        {chart.series[0].name}
      </p>
    );
  }

  return (
    <div className="mt-2 flex flex-wrap gap-2">
      {chart.series.map((series: ChartSeries, idx: number) => (
        <span key={series.name} className="inline-flex items-center gap-1.5 text-[11px] text-[#a3a3a3]">
          <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: CHART_COLORS[idx % CHART_COLORS.length] }} />
          {series.name}
        </span>
      ))}
    </div>
  );
}

function ComputedChartCard({ chart }: { chart: ComputedChart }) {
  const hasQuickChart = Boolean(chart.chart_url);
  const chartForSvg = normalizeChartForSvg(chart);

  return (
    <div
      className="rounded-[12px] border border-[#333333] bg-[#171717] p-4"
      data-testid={`computed-chart-${chart.chunk_id}`}
    >
      <div className="mb-3 flex flex-wrap items-start justify-between gap-2">
        <div>
          <p className="text-[12px] font-semibold uppercase tracking-[0.06em] text-[#a3a3a3]">
            {chart.title?.trim() || "Requested chart"}
          </p>
        </div>
        <p className="text-[11px] text-[#737373]">
          {chart.citation.filename} · p. {chart.citation.page_number} · chunk {chart.citation.chunk_id.slice(0, 8)}
        </p>
      </div>

      {hasQuickChart ? (
        <img
          src={chart.chart_url}
          alt={chart.title?.trim() || "Chart"}
          className="w-full max-w-[520px] rounded-[8px] border border-[#2a2a2a] bg-white"
          data-testid="quickchart-image"
        />
      ) : chart.chart_type === "line" ? (
        <LineChartSvg chart={chartForSvg} />
      ) : (
        <BarChartSvg chart={chartForSvg} />
      )}
      {chartForSvg.series.length > 0 ? <ChartLegend chart={chartForSvg} /> : null}
    </div>
  );
}

export function ComputedChartsPanel({ charts, isOpen, onToggleOpen, messageIndex }: ComputedChartsPanelProps) {
  if (charts.length === 0) return null;

  const chartLabel = charts.length === 1 ? "graph" : "graphs";

  return (
    <>
      <div className="w-full min-w-0" data-testid={`computed-charts-trigger-${messageIndex}`}>
        <button
          type="button"
          onClick={onToggleOpen}
          className="flex w-full items-center justify-between rounded-[10px] border border-[#2a2a2a] bg-gradient-to-b from-[#1a1a1a] to-[#141414] px-3.5 py-2.5 text-left transition-colors hover:border-[#404040]"
          data-testid={`toggle-charts-${messageIndex}`}
        >
          <span className="text-[12px] font-medium text-[#a3a3a3]">
            Chart · <span className="text-[#e5e5e5]">{charts.length}</span> {chartLabel}
          </span>
          <span className="text-[11px] font-medium text-[#d4d4d4]">{isOpen ? "Close" : "View graph"}</span>
        </button>
      </div>

      {isOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
          data-testid="computed-charts-overlay"
          role="dialog"
          aria-modal="true"
          aria-label="Chart viewer"
        >
          <button
            type="button"
            aria-label="Close chart"
            className="absolute inset-0 cursor-default"
            onClick={onToggleOpen}
          />
          <div
            className="relative z-10 flex max-h-[90vh] w-full max-w-3xl flex-col overflow-hidden rounded-[16px] border border-[#2a2a2a] bg-gradient-to-b from-[#1a1a1a] to-[#111111] shadow-2xl"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-[#2a2a2a] px-4 py-3">
              <span className="text-[13px] font-medium text-[#e5e5e5]">
                {charts.length === 1 ? "Chart" : `${charts.length} charts`}
              </span>
              <button
                type="button"
                onClick={onToggleOpen}
                className="rounded-[6px] border border-[#333333] px-2.5 py-1 text-[11px] font-medium text-[#d4d4d4] transition-colors hover:border-[#525252] hover:text-[#f5f5f5]"
                data-testid={`close-charts-${messageIndex}`}
              >
                Close
              </button>
            </div>
            <div className="space-y-3 overflow-y-auto p-3" data-testid="computed-charts-panel">
              {charts.map((chart) => (
                <ComputedChartCard key={chart.chunk_id} chart={chart} />
              ))}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
