interface ChatAvatarProps {
  role: "user" | "assistant";
}

function UserIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden>
      <circle cx="8" cy="5" r="2.6" stroke="currentColor" strokeWidth="1.4" />
      <path
        d="M2.8 13.2c0-2.4 2.3-3.9 5.2-3.9s5.2 1.5 5.2 3.9"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
      />
    </svg>
  );
}

function AssistantIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden>
      <rect x="3" y="4.5" width="10" height="7.5" rx="2.2" stroke="currentColor" strokeWidth="1.4" />
      <path d="M8 2.2V4.5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
      <circle cx="6" cy="8.2" r="0.95" fill="currentColor" />
      <circle cx="10" cy="8.2" r="0.95" fill="currentColor" />
    </svg>
  );
}

/** Small circular avatar shown beside each chat message. */
export function ChatAvatar({ role }: ChatAvatarProps) {
  const isUser = role === "user";
  return (
    <div
      aria-hidden
      className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full border ${
        isUser
          ? "border-[#404040] bg-gradient-to-br from-[#404040] to-[#2a2a2a] text-[#e5e5e5]"
          : "border-[#2a2a2a] bg-gradient-to-b from-[#1f1f1f] to-[#171717] text-[#a3a3a3]"
      }`}
    >
      {isUser ? <UserIcon /> : <AssistantIcon />}
    </div>
  );
}
