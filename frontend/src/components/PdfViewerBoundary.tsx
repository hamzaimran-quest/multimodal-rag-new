import { Component, type ReactNode } from "react";

import { pdfError } from "../pdf/log";

interface Props {
  children: ReactNode;
  onClose: () => void;
  /** Changing this key resets the boundary (e.g. when retargeting to a new doc). */
  resetKey?: string;
}

interface State {
  hasError: boolean;
  message: string;
}

/**
 * Isolates PDF viewer crashes. Without this, an exception thrown during render
 * (e.g. from a destroyed pdf.js document) propagates to the app root and blanks
 * the entire page. Here it degrades to a closable error panel instead.
 */
export class PdfViewerBoundary extends Component<Props, State> {
  state: State = { hasError: false, message: "" };

  static getDerivedStateFromError(error: unknown): State {
    return { hasError: true, message: error instanceof Error ? error.message : String(error) };
  }

  componentDidCatch(error: unknown): void {
    pdfError("viewer.crash", error);
  }

  componentDidUpdate(prevProps: Props): void {
    if (prevProps.resetKey !== this.props.resetKey && this.state.hasError) {
      this.setState({ hasError: false, message: "" });
    }
  }

  render(): ReactNode {
    if (this.state.hasError) {
      return (
        <div className="fixed right-0 top-0 z-40 flex h-[100dvh] h-screen w-[min(52vw,760px)] flex-col border-l border-[#2a2a2a] bg-[#0f0f0f] shadow-2xl shadow-black/60 max-[880px]:w-full">
          <div className="flex items-center justify-between border-b border-[#2a2a2a] bg-[#141414] px-4 py-3">
            <span className="text-[13.5px] font-semibold text-[#f5f5f5]">Document viewer</span>
            <button
              type="button"
              onClick={this.props.onClose}
              className="h-7 w-7 rounded-[6px] border border-[#333333] text-[#a3a3a3] hover:border-[#525252] hover:text-[#e5e5e5] max-[880px]:h-10 max-[880px]:w-10"
              aria-label="Close viewer"
            >
              ✕
            </button>
          </div>
          <div className="flex flex-1 items-center justify-center px-6 text-center text-[13px] text-rose-300">
            The document viewer hit an error and was closed to keep the app running.
            <br />
            {this.state.message}
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
