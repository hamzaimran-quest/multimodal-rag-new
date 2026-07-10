import { useLayoutEffect, useRef, useState } from "react";

import { deriveHeroImages } from "../lib/heroImages";
import type { ComputedChart, QuerySource } from "../types";
import { ComputedChartsPanel } from "./ComputedChartsPanel";
import { HeroImages } from "./HeroImages";
import { MarkdownAnswer } from "./MarkdownAnswer";
import { SourcesPanel } from "./SourcesPanel";

interface ChatAssistantMessageProps {
  messageIndex: number;
  text: string;
  sources: QuerySource[];
  charts: ComputedChart[];
  placeholder?: string;
  sourcesOpen: boolean;
  chartsOpen: boolean;
  onToggleSources: () => void;
  onToggleCharts: () => void;
  onGoToPage: () => void;
  onOpenSource: (source: QuerySource) => void;
}

export function ChatAssistantMessage({
  messageIndex,
  text,
  sources,
  charts,
  placeholder,
  sourcesOpen,
  chartsOpen,
  onToggleSources,
  onToggleCharts,
  onGoToPage,
  onOpenSource,
}: ChatAssistantMessageProps) {
  const bubbleRef = useRef<HTMLDivElement>(null);
  const [bubbleWidth, setBubbleWidth] = useState<number | null>(null);

  useLayoutEffect(() => {
    const element = bubbleRef.current;
    if (!element) return;

    const updateWidth = () => setBubbleWidth(element.offsetWidth);
    updateWidth();

    const observer = new ResizeObserver(updateWidth);
    observer.observe(element);
    return () => observer.disconnect();
  }, [text]);

  const hasExtras = sources.length > 0 || charts.length > 0;

  return (
    <div className="flex max-w-[82%] min-w-0 flex-col items-start max-[880px]:max-w-full">
      <div
        ref={bubbleRef}
        className="w-fit max-w-full rounded-[4px_16px_16px_16px] border border-[#2a2a2a] bg-gradient-to-b from-[#1f1f1f] to-[#171717] px-5 py-4 text-[15px] leading-[1.75] text-[#e5e5e5]"
        data-testid={`chat-msg-${messageIndex}`}
      >
        <MarkdownAnswer content={text} placeholder={placeholder} />
      </div>

      {sources.length > 0 && (
        <div className="mt-3 min-w-0 max-w-full overflow-hidden" style={bubbleWidth ? { width: bubbleWidth } : undefined}>
          <HeroImages images={deriveHeroImages(sources)} />
        </div>
      )}

      {hasExtras && (
        <div
          className="flex flex-col gap-2"
          style={bubbleWidth ? { width: bubbleWidth } : undefined}
        >
          {charts.length > 0 && (
            <ComputedChartsPanel
              charts={charts}
              isOpen={chartsOpen}
              onToggleOpen={onToggleCharts}
              messageIndex={messageIndex}
            />
          )}
          {sources.length > 0 && (
            <SourcesPanel
              sources={sources}
              isOpen={sourcesOpen}
              onToggleOpen={onToggleSources}
              messageIndex={messageIndex}
              onGoToPage={onGoToPage}
              onOpenSource={onOpenSource}
            />
          )}
        </div>
      )}
    </div>
  );
}
