import { create } from 'zustand';

export interface AlertData {
  alert_id: string;
  event_id: string;
  person_id: string;
  person_name: string;
  camera_id: string;
  camera_location: string;
  timestamp: string;
  similarity: number;
  severity_score: number;
  status: string;
  evidence_path?: string;
  operator_notes?: string;
}

interface AlertState {
  alerts: AlertData[];
  wsStatus: 'CONNECTED' | 'CONNECTING' | 'DISCONNECTED';
  unreadCount: number;
  addAlert: (alert: AlertData) => void;
  setWsStatus: (status: 'CONNECTED' | 'CONNECTING' | 'DISCONNECTED') => void;
  resetUnreadCount: () => void;
  clearAlerts: () => void;
}

export const useAlertStore = create<AlertState>((set) => ({
  alerts: [],
  wsStatus: 'DISCONNECTED',
  unreadCount: 0,

  addAlert: (alert) =>
    set((state) => {
      // Avoid duplicate alert inserts by checking alert_id
      const exists = state.alerts.some((a) => a.alert_id === alert.alert_id);
      if (exists) return {};

      // Insert new alert at the top, cap list at 50 to avoid memory growth
      const newAlerts = [alert, ...state.alerts].slice(0, 50);
      return {
        alerts: newAlerts,
        unreadCount: state.unreadCount + 1,
      };
    }),

  setWsStatus: (status) => set({ wsStatus: status }),

  resetUnreadCount: () => set({ unreadCount: 0 }),

  clearAlerts: () => set({ alerts: [], unreadCount: 0 }),
}));
