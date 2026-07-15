interface AssistantAvatarProps {
  className?: string;
}

/** Robot icon marking replies from the agent. */
function RobotIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" className="h-[60%] w-[60%]" aria-hidden>
      <circle cx="12" cy="3.1" r="1.5" stroke="currentColor" strokeWidth="1.4" />
      <line x1="12" y1="4.6" x2="12" y2="7" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
      <rect x="5" y="7" width="14" height="13" rx="2.6" stroke="currentColor" strokeWidth="1.4" />
      <rect x="2.2" y="11" width="2.8" height="6" rx="1" stroke="currentColor" strokeWidth="1.4" />
      <rect x="19" y="11" width="2.8" height="6" rx="1" stroke="currentColor" strokeWidth="1.4" />
      <circle cx="9.3" cy="14.5" r="1.3" stroke="currentColor" strokeWidth="1.4" />
      <circle cx="14.7" cy="14.5" r="1.3" stroke="currentColor" strokeWidth="1.4" />
    </svg>
  );
}

/** Brand mark reused from the sidebar logo, scaled down for message rows. */
export function AssistantAvatar({ className = "" }: AssistantAvatarProps) {
  return (
    <div
      className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-[9px] bg-gradient-to-br from-[#525252] via-[#404040] to-[#262626] text-[#e5e5e5] shadow-lg shadow-black/40 max-[880px]:h-7 max-[880px]:w-7 ${className}`}
      aria-hidden
      data-testid="assistant-avatar"
    >
      <RobotIcon />
    </div>
  );
}

interface UserAvatarProps {
  className?: string;
}

/** Person-in-circle icon marking messages sent by the user. */
function UserIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" className="h-[68%] w-[68%]" aria-hidden>
      <circle cx="12" cy="9.6" r="2.7" stroke="currentColor" strokeWidth="1.4" />
      <path d="M6.8 17.2C7.7 14.7 9.6 13.4 12 13.4s4.3 1.3 5.2 3.8" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  );
}

export function UserAvatar({ className = "" }: UserAvatarProps) {
  return (
    <div
      className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-[#404040] bg-gradient-to-br from-[#333333] to-[#1a1a1a] text-[#d4d4d4] max-[880px]:h-7 max-[880px]:w-7 ${className}`}
      aria-hidden
      data-testid="user-avatar"
    >
      <UserIcon />
    </div>
  );
}
