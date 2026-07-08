import type { IngestionStatus } from "../types";
import { estimateIngestionProgress } from "../utils/format";

interface IngestionProgressRingProps {
  status: IngestionStatus;
  uploadTimestamp: string | null;
  backendProgress: number;
  docId: string;
}

const ringClassByStatus: Record<IngestionStatus, string> = {
  pending: "text-[#d4d4d4]",
  processing: "text-[#d4d4d4]",
  indexed: "text-[#737373]",
  failed: "text-rose-300",
};

export function IngestionProgressRing({
  status,
  uploadTimestamp,
  backendProgress,
  docId,
}: IngestionProgressRingProps) {
  const estimated = estimateIngestionProgress(status, uploadTimestamp);
  const progress = Math.max(0, Math.min(100, backendProgress || estimated));
  const ringClass = ringClassByStatus[status];
  const isActive = status === "pending" || status === "processing";
  const label = status === "indexed" ? "done" : status === "failed" ? "!" : `${Math.round(progress)}%`;

  return (
    <div
      className="relative inline-flex h-10 w-10 items-center justify-center"
      data-testid={`progress-ring-${docId}`}
      aria-label={`ingestion-progress-${status}`}
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(progress)}
    >
      <svg
        className={`h-10 w-10 ${isActive ? "[animation:spin_1.6s_linear_infinite]" : ""}`}
        viewBox="0 0 36 36"
      >
        <path
          className="text-[#333333]"
          stroke="currentColor"
          strokeWidth="3.2"
          fill="none"
          d="M18 2.5a15.5 15.5 0 1 1 0 31a15.5 15.5 0 1 1 0-31"
        />
        <path
          className={ringClass}
          stroke="currentColor"
          strokeWidth="3.2"
          strokeLinecap="round"
          fill="none"
          strokeDasharray={`${(progress / 100) * 97.4} 97.4`}
          d="M18 2.5a15.5 15.5 0 1 1 0 31a15.5 15.5 0 1 1 0-31"
          transform="rotate(-90 18 18)"
        />
      </svg>
      <span className={`absolute text-[10px] font-semibold ${ringClass}`}>{label}</span>
    </div>
  );
}
