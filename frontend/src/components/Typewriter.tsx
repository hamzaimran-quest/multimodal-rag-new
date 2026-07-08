import { useEffect, useRef, useState } from "react";

interface TypewriterProps {
  /** Static text rendered before the animated phrase. */
  prefix?: string;
  /** Phrases that are typed, then erased, in a loop. */
  phrases: string[];
  className?: string;
  phraseClassName?: string;
  typingSpeedMs?: number;
  deletingSpeedMs?: number;
  holdMs?: number;
}

type Phase = "typing" | "holding" | "deleting";

export function Typewriter({
  prefix = "",
  phrases,
  className = "",
  phraseClassName = "",
  typingSpeedMs = 65,
  deletingSpeedMs = 32,
  holdMs = 1600,
}: TypewriterProps) {
  const [index, setIndex] = useState(0);
  const [text, setText] = useState("");
  const [phase, setPhase] = useState<Phase>("typing");
  const timeoutRef = useRef<number | null>(null);

  useEffect(() => {
    if (phrases.length === 0) return;
    const current = phrases[index % phrases.length];

    const schedule = (fn: () => void, delay: number) => {
      timeoutRef.current = window.setTimeout(fn, delay);
    };

    if (phase === "typing") {
      if (text.length < current.length) {
        schedule(() => setText(current.slice(0, text.length + 1)), typingSpeedMs);
      } else {
        schedule(() => setPhase("holding"), holdMs);
      }
    } else if (phase === "holding") {
      schedule(() => setPhase("deleting"), holdMs);
    } else {
      if (text.length > 0) {
        schedule(() => setText(current.slice(0, text.length - 1)), deletingSpeedMs);
      } else {
        setIndex((i) => (i + 1) % phrases.length);
        setPhase("typing");
      }
    }

    return () => {
      if (timeoutRef.current !== null) window.clearTimeout(timeoutRef.current);
    };
  }, [text, phase, index, phrases, typingSpeedMs, deletingSpeedMs, holdMs]);

  return (
    <span className={className} aria-live="polite">
      {prefix}
      <span className={phraseClassName}>{text}</span>
      <span className="ml-0.5 inline-block w-[2px] animate-[caret-blink_1s_infinite] bg-current align-middle" style={{ height: "1em" }} aria-hidden />
    </span>
  );
}
