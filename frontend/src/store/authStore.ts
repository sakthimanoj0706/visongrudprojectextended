import { create } from 'zustand';
import axios from 'axios';

interface AuthState {
  token: string | null;
  username: string | null;
  role: string | null;
  expireTime: number | null; // epoch seconds
  isAuthenticated: boolean;
  loading: boolean;
  error: string | null;
  login: (username: string, password: string, rememberMe?: boolean) => Promise<boolean>;
  logout: () => void;
  checkTokenExpiry: () => void;
  clearError: () => void;
}

// Helper to decode JWT token payload natively
const decodeToken = (token: string) => {
  try {
    const payloadBase64 = token.split('.')[1];
    const decodedJson = atob(payloadBase64);
    return JSON.parse(decodedJson);
  } catch (e) {
    console.error("Failed to decode JWT token:", e);
    return null;
  }
};

export const useAuthStore = create<AuthState>((set, get) => {
  // Load initial state from localStorage
  const savedToken = localStorage.getItem('vg_token') || sessionStorage.getItem('vg_token');
  let initialUser: string | null = null;
  let initialRole: string | null = null;
  let initialExp: number | null = null;
  let initialAuth = false;

  if (savedToken) {
    const decoded = decodeToken(savedToken);
    if (decoded && decoded.exp * 1000 > Date.now()) {
      initialUser = decoded.sub || null;
      initialRole = decoded.role || null;
      initialExp = decoded.exp || null;
      initialAuth = true;
    } else {
      localStorage.removeItem('vg_token');
      sessionStorage.removeItem('vg_token');
    }
  }

  return {
    token: savedToken || null,
    username: initialUser,
    role: initialRole,
    expireTime: initialExp,
    isAuthenticated: initialAuth,
    loading: false,
    error: null,

    login: async (username, password, rememberMe = false) => {
      set({ loading: true, error: null });
      try {
        const response = await axios.post('/api/v1/auth/token', {
          username,
          password
        });
        
        const { access_token } = response.data;
        const decoded = decodeToken(access_token);
        
        if (!decoded) {
          throw new Error("Invalid server token payload");
        }

        const role = decoded.role || "Investigator"; // fallback role
        const exp = decoded.exp || Math.floor((Date.now() / 1000) + 3600); // fallback 1 hour

        // Save token based on Remember Me
        if (rememberMe) {
          localStorage.setItem('vg_token', access_token);
        } else {
          sessionStorage.setItem('vg_token', access_token);
        }

        set({
          token: access_token,
          username: decoded.sub || username,
          role,
          expireTime: exp,
          isAuthenticated: true,
          loading: false,
          error: null
        });

        return true;
      } catch (err: any) {
        let msg = "Authentication failed. Connection error.";
        if (err.response && err.response.data && err.response.data.detail) {
          msg = err.response.data.detail;
        } else if (err.response && err.response.status === 401) {
          msg = "Incorrect username or password.";
        }
        set({ error: msg, loading: false });
        return false;
      }
    },

    logout: () => {
      localStorage.removeItem('vg_token');
      sessionStorage.removeItem('vg_token');
      set({
        token: null,
        username: null,
        role: null,
        expireTime: null,
        isAuthenticated: false,
        error: null
      });
    },

    checkTokenExpiry: () => {
      const { expireTime, isAuthenticated, logout } = get();
      if (isAuthenticated && expireTime) {
        // expireTime is in seconds, Date.now() is in milliseconds
        if (expireTime * 1000 < Date.now()) {
          logout();
        }
      }
    },

    clearError: () => set({ error: null })
  };
});
