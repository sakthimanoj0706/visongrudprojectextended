import { useAlertStore } from '../store/alertStore';
import type { AlertData } from '../store/alertStore';

let socket: WebSocket | null = null;
let reconnectDelay = 2000;
let maxReconnectDelay = 30000;
let reconnectTimer: any = null;

const getWsUrl = (): string => {
  const apiUrl = import.meta.env.VITE_API_URL;
  if (apiUrl) {
    // If absolute API URL exists, replace http/https with ws/wss
    const url = new URL(apiUrl);
    const protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${url.host}/api/v1/alerts/ws`;
  }
  // Fallback to relative path which will go through Vite dev proxy
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}/ws/api/v1/alerts/ws`;
};

export const connectWebSocket = () => {
  const alertStore = useAlertStore.getState();
  
  // Prevent duplicate connections
  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
    return;
  }

  // Clear any existing reconnect timers
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }

  const wsUrl = getWsUrl();
  console.log(`[WEBSOCKET] Connecting to ${wsUrl}...`);
  alertStore.setWsStatus('CONNECTING');

  try {
    socket = new WebSocket(wsUrl);

    socket.onopen = () => {
      console.log('[WEBSOCKET] Connected successfully.');
      alertStore.setWsStatus('CONNECTED');
      reconnectDelay = 2000; // Reset reconnect delay on successful connection
    };

    socket.onmessage = (event) => {
      try {
        const rawData = JSON.parse(event.data);
        console.log('[WEBSOCKET] Alert received:', rawData);
        
        // Map alert payload details
        const alert: AlertData = {
          alert_id: rawData.alert_id,
          event_id: rawData.event_id,
          person_id: rawData.person_id || 'unregistered',
          person_name: rawData.person_name || 'Unknown Target',
          camera_id: rawData.camera_id || 'CAM',
          camera_location: rawData.camera_location || 'Unknown Location',
          timestamp: rawData.timestamp || new Date().toISOString(),
          similarity: rawData.similarity || 0.0,
          severity_score: rawData.severity_score || 0.5,
          status: rawData.status || 'ACTIVE',
          evidence_path: rawData.evidence_path || undefined
        };

        useAlertStore.getState().addAlert(alert);
      } catch (e) {
        console.error('[WEBSOCKET] Error parsing message payload:', e);
      }
    };

    socket.onclose = (event) => {
      console.log(`[WEBSOCKET] Connection closed: ${event.reason} (code ${event.code})`);
      alertStore.setWsStatus('DISCONNECTED');
      scheduleReconnect();
    };

    socket.onerror = (error) => {
      console.error('[WEBSOCKET] Error occurred:', error);
      alertStore.setWsStatus('DISCONNECTED');
    };

  } catch (err) {
    console.error('[WEBSOCKET] Setup error:', err);
    alertStore.setWsStatus('DISCONNECTED');
    scheduleReconnect();
  }
};

const scheduleReconnect = () => {
  if (reconnectTimer) return;
  
  console.log(`[WEBSOCKET] Reconnecting in ${(reconnectDelay / 1000).toFixed(1)}s...`);
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    // Exponential backoff
    reconnectDelay = Math.min(reconnectDelay * 1.5, maxReconnectDelay);
    connectWebSocket();
  }, reconnectDelay);
};

export const disconnectWebSocket = () => {
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
  if (socket) {
    // Remove listeners before closing to prevent auto-reconnect loops
    socket.onopen = null;
    socket.onmessage = null;
    socket.onclose = null;
    socket.onerror = null;
    socket.close();
    socket = null;
  }
  useAlertStore.getState().setWsStatus('DISCONNECTED');
};
