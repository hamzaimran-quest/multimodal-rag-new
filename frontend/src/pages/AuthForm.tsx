import { useState } from "react";

interface AuthFormProps {
  mode: "login" | "signup";
  onSubmit: (email: string, password: string) => Promise<void>;
  error: string | null;
  onSwitch: () => void;
}

const COPY = {
  login: {
    title: "Welcome back",
    subtitle: "Sign in to your document workspace",
    submit: "Sign in",
    switchPrompt: "Don't have an account?",
    switchAction: "Create one",
  },
  signup: {
    title: "Create your account",
    subtitle: "Each account keeps its own private set of documents",
    submit: "Sign up",
    switchPrompt: "Already have an account?",
    switchAction: "Sign in",
  },
} as const;

export function AuthForm({ mode, onSubmit, error, onSwitch }: AuthFormProps) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const copy = COPY[mode];
  const canSubmit = email.trim().length > 0 && password.length > 0 && !submitting;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!canSubmit) return;
    setSubmitting(true);
    try {
      await onSubmit(email.trim(), password);
    } finally {
      setSubmitting(false);
    }
  };

  const inputClass =
    "w-full rounded-[13px] border border-[#333333] bg-gradient-to-b from-[#1c1c1c] to-[#161616] px-4 py-4 text-[15.5px] text-[#f5f5f5] outline-none transition-all duration-150 placeholder:text-[#5a5a5a] hover:border-[#404040] focus:border-[#6a6a6a] focus:from-[#202020] focus:to-[#181818] focus:ring-2 focus:ring-[#404040]/40";

  return (
    <div className="w-full max-w-[440px]">
      <div className="mb-8 flex items-center gap-3">
        <div className="flex h-[42px] w-[42px] items-center justify-center rounded-[12px] bg-gradient-to-br from-[#525252] via-[#404040] to-[#262626] font-['Space_Grotesk'] text-xl font-bold text-[#e5e5e5] shadow-lg shadow-black/40">
          M
        </div>
        <div>
          <h1 className="font-['Space_Grotesk'] text-[19px] font-semibold text-[#f5f5f5]">Multimodal RAG</h1>
          <p className="text-[11.5px] text-[#a3a3a3]">Document Q&amp;A</p>
        </div>
      </div>

      <div className="auth-fade-in">
        <h2 className="font-['Space_Grotesk'] text-[28px] font-semibold leading-tight text-[#f5f5f5]">{copy.title}</h2>
        <p className="mt-2 text-[14.5px] text-[#a3a3a3]">{copy.subtitle}</p>

        <form className="mt-8 space-y-5" onSubmit={handleSubmit}>
          <div>
            <label className="mb-2 block text-[13px] font-medium text-[#c4c4c4]" htmlFor="email">
              Email
            </label>
            <input
              id="email"
              type="email"
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              className={inputClass}
              data-testid="auth-email"
            />
          </div>
          <div>
            <label className="mb-2 block text-[13px] font-medium text-[#c4c4c4]" htmlFor="password">
              Password
            </label>
            <input
              id="password"
              type="password"
              autoComplete={mode === "login" ? "current-password" : "new-password"}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder={mode === "signup" ? "At least 8 characters" : "Your password"}
              className={inputClass}
              data-testid="auth-password"
            />
          </div>

          {error && (
            <div
              className="rounded-[10px] border border-rose-500/35 bg-rose-500/10 px-4 py-3 text-[13.5px] text-rose-200"
              data-testid="auth-error"
            >
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={!canSubmit}
            className="mt-1 w-full rounded-[13px] bg-gradient-to-b from-[#5c5c5c] to-[#363636] px-4 py-4 text-[15.5px] font-semibold text-[#f5f5f5] shadow-lg shadow-black/40 transition-all duration-150 hover:from-[#737373] hover:to-[#454545] active:scale-[0.99] disabled:opacity-40 disabled:hover:from-[#5c5c5c] disabled:hover:to-[#363636]"
            data-testid="auth-submit"
          >
            {submitting ? "Please wait..." : copy.submit}
          </button>
        </form>

        <p className="mt-7 text-center text-[13.5px] text-[#a3a3a3]">
          {copy.switchPrompt}{" "}
          <button
            type="button"
            onClick={onSwitch}
            className="font-semibold text-[#e5e5e5] hover:underline"
            data-testid="auth-switch"
          >
            {copy.switchAction}
          </button>
        </p>
      </div>
    </div>
  );
}
