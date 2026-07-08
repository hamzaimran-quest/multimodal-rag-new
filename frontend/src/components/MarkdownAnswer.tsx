import { useMemo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface MarkdownAnswerProps {
  content: string;
  placeholder?: string;
}

/** A GFM delimiter row, e.g. `| --- | :--: |` — every cell is only dashes/colons. */
function isTableDelimiterRow(line: string): boolean {
  const trimmed = line.trim();
  if (!trimmed.includes("|") || !trimmed.includes("-")) return false;
  return trimmed
    .replace(/^\||\|$/g, "")
    .split("|")
    .every((cell) => /^\s*:?-+:?\s*$/.test(cell));
}

/**
 * LLMs frequently emit a table glued to the preceding line (often a list item)
 * without the blank line GFM requires to start a table block. Without it the
 * pipe rows render as inline text. Insert the missing blank line before any
 * header row that is immediately followed by a delimiter row.
 */
function ensureTableBlankLines(markdown: string): string {
  const lines = markdown.split("\n");
  const out: string[] = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const next = lines[i + 1] ?? "";
    const startsTable = line.includes("|") && !isTableDelimiterRow(line) && isTableDelimiterRow(next);
    if (startsTable) {
      const prev = out.length ? out[out.length - 1] : "";
      if (prev.trim() !== "") out.push("");
    }
    out.push(line);
  }
  return out.join("\n");
}

export function MarkdownAnswer({ content, placeholder = "..." }: MarkdownAnswerProps) {
  const normalized = useMemo(() => ensureTableBlankLines(content), [content]);

  if (!content) {
    return <span className="text-[#737373]">{placeholder}</span>;
  }

  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        h2: ({ children }) => (
          <h2 className="mb-2 mt-4 font-['Space_Grotesk'] text-base font-semibold text-[#f5f5f5] first:mt-0">
            {children}
          </h2>
        ),
        h3: ({ children }) => (
          <h3 className="mb-1.5 mt-3 text-sm font-semibold text-[#e5e5e5]">{children}</h3>
        ),
        p: ({ children }) => <p className="mb-3 last:mb-0">{children}</p>,
        ul: ({ children }) => <ul className="mb-3 list-disc space-y-1.5 pl-5">{children}</ul>,
        ol: ({ children }) => <ol className="mb-3 list-decimal space-y-1.5 pl-5">{children}</ol>,
        li: ({ children }) => <li className="leading-relaxed">{children}</li>,
        strong: ({ children }) => <strong className="font-semibold text-[#f5f5f5]">{children}</strong>,
        em: ({ children }) => <em className="text-[#d4d4d4]">{children}</em>,
        table: ({ children }) => (
          <div className="mb-3 overflow-x-auto rounded-[8px] border border-[#333333]">
            <table className="w-full min-w-[280px] border-collapse text-sm">{children}</table>
          </div>
        ),
        thead: ({ children }) => <thead className="bg-[#262626]">{children}</thead>,
        th: ({ children }) => (
          <th className="border-b border-[#333333] px-3 py-2 text-left font-semibold text-[#f5f5f5]">
            {children}
          </th>
        ),
        td: ({ children }) => (
          <td className="border-b border-[#2a2a2a] px-3 py-2 align-top text-[#e5e5e5]">{children}</td>
        ),
        tr: ({ children }) => <tr className="even:bg-[#1a1a1a]/50">{children}</tr>,
        blockquote: ({ children }) => (
          <blockquote className="mb-3 border-l-2 border-[#525252] pl-3 text-[#a3a3a3]">{children}</blockquote>
        ),
        code: ({ children }) => (
          <code className="rounded bg-[#262626] px-1.5 py-0.5 font-['JetBrains_Mono'] text-[13px] text-[#d4d4d4]">
            {children}
          </code>
        ),
      }}
    >
      {normalized}
    </ReactMarkdown>
  );
}
