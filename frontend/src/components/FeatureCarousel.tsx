import { useEffect, useState } from "react";

interface Feature {
  title: string;
  description: string;
  icon: React.ReactNode;
}

const FEATURES: Feature[] = [
  {
    title: "Grounded answers",
    description: "Every response is generated only from your own documents — no hallucinated numbers.",
    icon: (
      <path d="M4 5h16M4 10h16M4 15h10" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    ),
  },
  {
    title: "Cited to the source",
    description: "Inspect the exact page, table, or chart behind each answer in the Sources panel.",
    icon: (
      <>
        <rect x="4" y="3" width="16" height="18" rx="2" stroke="currentColor" strokeWidth="1.6" />
        <path d="M8 8h8M8 12h8M8 16h5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      </>
    ),
  },
  {
    title: "Charts, extracted",
    description: "Tables become interactive charts so you can compare metrics across fiscal years.",
    icon: (
      <path d="M5 19V9M12 19V5M19 19v-7" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    ),
  },
  {
    title: "Private by default",
    description: "Your workspace is isolated — nobody else can see or search your uploads.",
    icon: (
      <>
        <rect x="5" y="10" width="14" height="10" rx="2" stroke="currentColor" strokeWidth="1.6" />
        <path d="M8 10V7a4 4 0 0 1 8 0v3" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      </>
    ),
  },
];

export function FeatureCarousel() {
  const [active, setActive] = useState(0);

  useEffect(() => {
    const id = window.setInterval(() => {
      setActive((i) => (i + 1) % FEATURES.length);
    }, 4000);
    return () => window.clearInterval(id);
  }, []);

  const feature = FEATURES[active];

  return (
    <div className="w-full">
      <div
        key={active}
        className="auth-fade-in rounded-[20px] border border-[#2a2a2a] bg-gradient-to-b from-[#161616]/90 to-[#0f0f0f]/90 p-9 shadow-2xl shadow-black/40 backdrop-blur-sm"
      >
        <div className="mb-6 flex h-[60px] w-[60px] items-center justify-center rounded-[16px] bg-gradient-to-br from-[#3a3a3a] to-[#1f1f1f] text-[#e5e5e5] shadow-lg shadow-black/40">
          <svg width="30" height="30" viewBox="0 0 24 24" fill="none">
            {feature.icon}
          </svg>
        </div>
        <h3 className="font-['Space_Grotesk'] text-[22px] font-semibold text-[#f5f5f5]">{feature.title}</h3>
        <p className="mt-3 text-[15px] leading-relaxed text-[#a3a3a3]">{feature.description}</p>
      </div>

      <div className="mt-6 flex items-center justify-center gap-2">
        {FEATURES.map((f, i) => (
          <button
            key={f.title}
            type="button"
            onClick={() => setActive(i)}
            aria-label={`Show ${f.title}`}
            className={`h-1.5 rounded-full transition-all duration-300 ${
              i === active ? "w-6 bg-[#a3a3a3]" : "w-1.5 bg-[#3a3a3a] hover:bg-[#525252]"
            }`}
          />
        ))}
      </div>
    </div>
  );
}
