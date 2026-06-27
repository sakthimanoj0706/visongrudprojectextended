import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import {
  Box,
  Button,
  Checkbox,
  FormControlLabel,
  IconButton,
  InputAdornment,
  TextField,
  Typography,
  Alert,
  CircularProgress,
  Card,
  Divider
} from '@mui/material';
import { Visibility, VisibilityOff, Wifi, WifiOff } from '@mui/icons-material';
import axios from 'axios';

import { useAuthStore } from '../store/authStore';

export const Login: React.FC = () => {
  const navigate = useNavigate();
  const login = useAuthStore((state) => state.login);
  const error = useAuthStore((state) => state.error);
  const loading = useAuthStore((state) => state.loading);
  const clearError = useAuthStore((state) => state.clearError);
  const isAuthenticated = useAuthStore((state) => state.isAuthenticated);

  const [showPassword, setShowPassword] = useState(false);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [rememberMe, setRememberMe] = useState(false);

  const { register, handleSubmit, formState: { errors } } = useForm();

  // Redirect to dashboard if already authenticated
  useEffect(() => {
    if (isAuthenticated) {
      navigate('/dashboard', { replace: true });
    }
  }, [isAuthenticated, navigate]);

  // Check backend server status on page load
  useEffect(() => {
    clearError();
    const checkBackend = async () => {
      try {
        await axios.get('/openapi.json', { timeout: 3000 });
        setBackendOnline(true);
      } catch (e) {
        setBackendOnline(false);
      }
    };
    checkBackend();
  }, [clearError]);

  const handleTogglePassword = () => setShowPassword(!showPassword);

  const onSubmit = async (data: any) => {
    const success = await login(data.username, data.password, rememberMe);
    if (success) {
      navigate('/dashboard', { replace: true });
    }
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'radial-gradient(circle at 50% 50%, #0C152E 0%, #05070D 100%)',
        p: 2
      }}
    >
      <Card
        sx={{
          width: '100%',
          maxWidth: 420,
          p: 4,
          textAlign: 'center',
        }}
      >
        {/* Logo and Platform Header */}
        <Box sx={{ mb: 4, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <Box
            sx={{
              width: 48,
              height: 48,
              borderRadius: '12px',
              background: 'linear-gradient(135deg, #00A8FF 0%, #0075B3 100%)',
              mb: 2,
              boxShadow: '0 0 20px rgba(0, 168, 255, 0.4)'
            }}
          />
          <Typography variant="h4" sx={{ fontWeight: 800, letterSpacing: '-0.03em', color: 'primary.main', mb: 0.5 }}>
            VisionGuard
          </Typography>
          <Typography variant="body2" sx={{ color: 'text.secondary', fontWeight: 500 }}>
            AI Surveillance Intelligence Platform
          </Typography>
        </Box>

        {/* Error Alert Banner */}
        {error && (
          <Alert severity="error" sx={{ mb: 3, borderRadius: '8px', textAlign: 'left' }} onClose={clearError}>
            {error}
          </Alert>
        )}

        {/* Login Form */}
        <Box component="form" onSubmit={handleSubmit(onSubmit)} noValidate>
          <TextField
            margin="normal"
            required
            fullWidth
            id="username"
            label="Username"
            autoComplete="username"
            autoFocus
            {...register('username', { required: 'Username is required' })}
            error={Boolean(errors.username)}
            helperText={errors.username?.message as string}
            sx={{
              '& .MuiOutlinedInput-root': {
                bgcolor: 'rgba(5, 7, 13, 0.4)',
                borderRadius: '8px',
              }
            }}
          />
          <TextField
            margin="normal"
            required
            fullWidth
            label="Password"
            type={showPassword ? 'text' : 'password'}
            id="password"
            autoComplete="current-password"
            {...register('password', { required: 'Password is required' })}
            error={Boolean(errors.password)}
            helperText={errors.password?.message as string}
            slotProps={{
              input: {
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton onClick={handleTogglePassword} edge="end">
                      {showPassword ? <VisibilityOff /> : <Visibility />}
                    </IconButton>
                  </InputAdornment>
                ),
              }
            }}
            sx={{
              '& .MuiOutlinedInput-root': {
                bgcolor: 'rgba(5, 7, 13, 0.4)',
                borderRadius: '8px',
              }
            }}
          />

          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 2, mb: 3 }}>
            <FormControlLabel
              control={
                <Checkbox
                  value="remember"
                  color="primary"
                  checked={rememberMe}
                  onChange={(e) => setRememberMe(e.target.checked)}
                />
              }
              label={<Typography variant="body2">Remember Me</Typography>}
            />
            <Typography
              variant="body2"
              color="primary"
              sx={{ cursor: 'pointer', '&:hover': { textDecoration: 'underline' } }}
            >
              Forgot Password?
            </Typography>
          </Box>

          <Button
            type="submit"
            fullWidth
            variant="contained"
            disabled={loading}
            sx={{
              py: 1.25,
              fontSize: '15px',
              mb: 3
            }}
          >
            {loading ? <CircularProgress size={24} color="inherit" /> : 'LOGIN'}
          </Button>

          {/* Backend Connection status */}
          <Divider sx={{ borderColor: 'rgba(148, 163, 184, 0.12)', mb: 2 }} />
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="body2" sx={{ color: 'text.secondary', fontSize: '11px' }}>
              System Version: v1.0.0
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              {backendOnline === null ? (
                <CircularProgress size={12} color="inherit" />
              ) : backendOnline ? (
                <>
                  <Wifi color="success" sx={{ fontSize: 14 }} />
                  <Typography variant="body2" sx={{ color: 'success.main', fontSize: '11px', fontWeight: 600 }}>
                    Backend: ONLINE
                  </Typography>
                </>
              ) : (
                <>
                  <WifiOff color="error" sx={{ fontSize: 14 }} />
                  <Typography variant="body2" sx={{ color: 'error.main', fontSize: '11px', fontWeight: 600 }}>
                    Backend: OFFLINE
                  </Typography>
                </>
              )}
            </Box>
          </Box>
        </Box>
      </Card>
    </Box>
  );
};

export default Login;
