import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import ProtectedRoute from '../components/ProtectedRoute';
import Login from '../pages/Login';
import Dashboard from '../pages/Dashboard';
import MainLayout from '../layouts/MainLayout';

export const AppRoutes: React.FC = () => {
  return (
    <Routes>
      {/* Public Route: Login */}
      <Route path="/login" element={<Login />} />

      {/* Protected Routes inside Main SOC Layout */}
      <Route
        path="/*"
        element={
          <ProtectedRoute>
            <MainLayout />
          </ProtectedRoute>
        }
      >
        {/* Child Pages inside MainLayout grid */}
        <Route path="dashboard" element={<Dashboard />} />
        
        {/* Placeholders for future phases */}
        <Route path="streams" element={<div style={{ padding: 24 }}><h2>Live Streams (Phase 10.2)</h2></div>} />
        <Route path="watchlist" element={<div style={{ padding: 24 }}><h2>Watchlist Registry (Phase 10.2)</h2></div>} />
        <Route path="tracking" element={<div style={{ padding: 24 }}><h2>Multi-Camera Tracking (Phase 10.3)</h2></div>} />
        <Route path="alerts" element={<div style={{ padding: 24 }}><h2>Alert Management (Phase 10.4)</h2></div>} />
        <Route path="assistant" element={<div style={{ padding: 24 }}><h2>NL Investigation Assistant (Phase 10.4)</h2></div>} />
        <Route path="evidence" element={<div style={{ padding: 24 }}><h2>Evidence Archive</h2></div>} />
        <Route path="analytics" element={<div style={{ padding: 24 }}><h2>Forensic Analytics</h2></div>} />
        <Route path="cameras" element={<div style={{ padding: 24 }}><h2>Camera Management</h2></div>} />
        <Route path="users" element={<div style={{ padding: 24 }}><h2>User Management</h2></div>} />
        <Route path="settings" element={<div style={{ padding: 24 }}><h2>Settings</h2></div>} />

        {/* Redirect root wildcard to dashboard */}
        <Route path="" element={<Navigate to="dashboard" replace />} />
        <Route path="*" element={<Navigate to="dashboard" replace />} />
      </Route>
    </Routes>
  );
};

export default AppRoutes;
