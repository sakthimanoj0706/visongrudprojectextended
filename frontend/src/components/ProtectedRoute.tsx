import React from 'react';
import { Navigate } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';

interface ProtectedRouteProps {
  children: React.ReactNode;
}

export const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ children }) => {
  const isAuthenticated = useAuthStore((state) => state.isAuthenticated);

  if (!isAuthenticated) {
    // Redirect to login page if unauthenticated
    return <Navigate to="/login" replace />;
  }

  // Render dashboard layout and components if authenticated
  return <>{children}</>;
};
export default ProtectedRoute;
