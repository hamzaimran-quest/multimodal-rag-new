import { useState } from "react";
import { Navigate, useNavigate } from "react-router-dom";

import { useAuth } from "../auth/AuthContext";
import { Typewriter } from "../components/Typewriter";
import { AuthForm } from "./AuthForm";

export function LoginPage() {
  const { user, initializing, login } = useAuth();
  const navigate = useNavigate();
  const [error, setError] = useState<string | null>(null);

  if (!initializing && user) return <Navigate to="/" replace />;

  const handleSubmit = async (email: string, password: string) => {
    setError(null);
    try {
      await login(email, password);
      navigate("/", { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    }
  };

  return (
    <div className="relative grid min-h-screen min-h-[100dvh] grid-cols-1 md:grid-cols-2">
      {/* Gradient separator down the seam */}
      <div className="pointer-events-none absolute inset-y-0 left-1/2 hidden w-[2px] -translate-x-1/2 bg-gradient-to-b from-transparent via-[#6a6a6a]/80 to-transparent shadow-[0_0_12px_rgba(120,120,120,0.35)] md:block" />

      {/* Left: animated headline — same fading gradient, mirrored toward the seam */}
      <div className="relative hidden overflow-hidden bg-[radial-gradient(ellipse_75%_85%_at_100%_50%,rgba(120,120,120,0.14),transparent_60%),linear-gradient(to_left,#131316,#0d0d0f_55%,#0a0a0a)] md:flex md:flex-col md:justify-center md:px-[6%] lg:px-[8%]">
        <div className="auth-blob pointer-events-none absolute -left-16 top-24 h-64 w-64 rounded-full bg-[#525252]/10 blur-3xl" />
        <div className="auth-blob pointer-events-none absolute bottom-16 right-8 h-72 w-72 rounded-full bg-[#3a3a4a]/10 blur-3xl" style={{ animationDelay: "2s" }} />

        <div className="relative z-10 w-full max-w-[620px]">
          <p className="mb-5 text-[13px] font-semibold uppercase tracking-[0.18em] text-[#737373]">
            Multimodal RAG
          </p>
          <h1 className="font-['Space_Grotesk'] text-[44px] font-semibold leading-[1.15] text-[#f5f5f5] lg:text-[52px]">
            <Typewriter
              prefix="A smarter way to "
              phrases={[
                "search any document.",
                "cite every answer.",
                "chat with your files.",
                "compare data across sources.",
              ]}
              phraseClassName="bg-gradient-to-r from-[#e5e5e5] to-[#737373] bg-clip-text text-transparent"
            />
          </h1>
          <p className="mt-7 max-w-[540px] text-[16px] leading-relaxed text-[#a3a3a3]">
            Ask questions in plain language and get grounded answers, cited to the exact page,
            table, and chart in your documents.
          </p>
        </div>
      </div>

      {/* Right: sign-in form — background fades in from the aurora panel on the left */}
      <div className="flex items-center justify-center bg-[radial-gradient(ellipse_75%_85%_at_0%_50%,rgba(120,120,120,0.14),transparent_60%),linear-gradient(to_right,#131316,#0d0d0f_55%,#0a0a0a)] px-8 py-12 md:px-14">
        <AuthForm mode="login" onSubmit={handleSubmit} error={error} onSwitch={() => navigate("/signup")} />
      </div>
    </div>
  );
}
