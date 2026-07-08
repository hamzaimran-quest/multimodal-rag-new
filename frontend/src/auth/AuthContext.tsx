import { createContext, useContext, useEffect, useMemo, useState } from "react";

import {
  loginRequest,
  logoutRequest,
  refreshRequest,
  signupRequest,
  type AuthUser,
} from "../api/auth";
import { setAccessToken } from "./tokenStore";

interface AuthContextValue {
  user: AuthUser | null;
  initializing: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [initializing, setInitializing] = useState(true);

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const result = await refreshRequest();
        if (!active) return;
        setAccessToken(result.access_token);
        setUser(result.user);
      } catch {
        if (!active) return;
        setAccessToken(null);
        setUser(null);
      } finally {
        if (active) setInitializing(false);
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      user,
      initializing,
      login: async (email, password) => {
        const result = await loginRequest(email, password);
        setAccessToken(result.access_token);
        setUser(result.user);
      },
      signup: async (email, password) => {
        const result = await signupRequest(email, password);
        setAccessToken(result.access_token);
        setUser(result.user);
      },
      logout: async () => {
        try {
          await logoutRequest();
        } finally {
          setAccessToken(null);
          setUser(null);
        }
      },
    }),
    [user, initializing],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (ctx === null) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}
