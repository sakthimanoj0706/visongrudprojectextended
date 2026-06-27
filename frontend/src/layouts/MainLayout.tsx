import React, { useState, useEffect } from 'react';
import { Outlet, useNavigate, useLocation, Link } from 'react-router-dom';
import {
  Box,
  Drawer,
  AppBar,
  Toolbar,
  List,
  Typography,
  Divider,
  IconButton,
  Badge,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  ListItemButton,
  Chip,
  Tooltip
} from '@mui/material';
import {
  Menu as MenuIcon,
  ChevronLeft as ChevronLeftIcon,
  Dashboard as DashboardIcon,
  Videocam as VideocamIcon,
  People as PeopleIcon,
  TrackChanges as TrackChangesIcon,
  NotificationsActive as AlertsIcon,
  Forum as AssistantIcon,
  FolderOpen as EvidenceIcon,
  BarChart as AnalyticsIcon,
  CameraAlt as CameraMgmtIcon,
  SupervisorAccount as UserMgmtIcon,
  Settings as SettingsIcon,
  AccountCircle,
  PowerSettingsNew as LogoutIcon,
  OfflineBolt as GpuIcon,
  CellTower as WsIcon,
  Storage as DbIcon
} from '@mui/icons-material';

import { useAuthStore } from '../store/authStore';
import { useAlertStore } from '../store/alertStore';
import { connectWebSocket, disconnectWebSocket } from '../websocket/wsClient';

const drawerWidth = 240;

export const MainLayout: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const logout = useAuthStore((state) => state.logout);
  const username = useAuthStore((state) => state.username);
  const role = useAuthStore((state) => state.role);
  
  const wsStatus = useAlertStore((state) => state.wsStatus);
  const unreadCount = useAlertStore((state) => state.unreadCount);
  const resetUnreadCount = useAlertStore((state) => state.resetUnreadCount);

  const [open, setOpen] = useState(true);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);

  // Initialize WebSocket Alert Feed on login/layout mount
  useEffect(() => {
    connectWebSocket();
    return () => {
      disconnectWebSocket();
    };
  }, []);

  const handleDrawerOpen = () => setOpen(true);
  const handleDrawerClose = () => setOpen(false);

  const handleProfileMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => setAnchorEl(null);

  const handleLogout = () => {
    handleMenuClose();
    logout();
    navigate('/login');
  };

  const menuItems = [
    { text: 'Dashboard', icon: <DashboardIcon />, path: '/dashboard' },
    { text: 'Live Surveillance', icon: <VideocamIcon />, path: '/streams' },
    { text: 'Watchlist Registry', icon: <PeopleIcon />, path: '/watchlist' },
    { text: 'Multi-Camera Tracking', icon: <TrackChangesIcon />, path: '/tracking' },
    { text: 'Alerts Management', icon: <AlertsIcon />, path: '/alerts' },
    { text: 'Investigation Assistant', icon: <AssistantIcon />, path: '/assistant' },
    { text: 'Evidence Archive', icon: <EvidenceIcon />, path: '/evidence' },
    { text: 'System Analytics', icon: <AnalyticsIcon />, path: '/analytics' },
    { text: 'Camera Configurations', icon: <CameraMgmtIcon />, path: '/cameras' },
    { text: 'User Configurations', icon: <UserMgmtIcon />, path: '/users' },
    { text: 'Global Settings', icon: <SettingsIcon />, path: '/settings' },
  ];

  // Helper to map route paths to page titles
  const getPageTitle = () => {
    const currentItem = menuItems.find((item) => item.path === location.pathname);
    return currentItem ? currentItem.text : 'Operations Center';
  };

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', bgcolor: 'background.default' }}>
      
      {/* Top Navbar */}
      <AppBar
        position="fixed"
        sx={{
          zIndex: (theme) => theme.zIndex.drawer + 1,
          transition: (theme) =>
            theme.transitions.create(['width', 'margin'], {
              easing: theme.transitions.easing.sharp,
              duration: theme.transitions.duration.leavingScreen,
            }),
          ...(open && {
            marginLeft: `${drawerWidth}px`,
            width: `calc(100% - ${drawerWidth}px)`,
            transition: (theme) =>
              theme.transitions.create(['width', 'margin'], {
                easing: theme.transitions.easing.sharp,
                duration: theme.transitions.duration.enteringScreen,
              }),
          }),
          backgroundColor: 'rgba(5, 7, 13, 0.8)',
          backdropFilter: 'blur(8px)',
          borderBottom: '1px solid rgba(26, 37, 60, 0.4)',
          boxShadow: 'none',
        }}
      >
        <Toolbar sx={{ justifyContent: 'space-between', px: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <IconButton
              color="inherit"
              aria-label="open drawer"
              onClick={handleDrawerOpen}
              edge="start"
              sx={{ marginRight: 2, ...(open && { display: 'none' }) }}
            >
              <MenuIcon />
            </IconButton>
            <Typography variant="h6" noWrap component="div" sx={{ fontWeight: 600, letterSpacing: '-0.02em' }}>
              {getPageTitle()}
            </Typography>
          </Box>

          {/* Diagnostic status lights and User Controls */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            
            {/* Realtime Status indicators */}
            <Box sx={{ display: { xs: 'none', md: 'flex' }, alignItems: 'center', gap: 1.5 }}>
              <Tooltip title={`WebSocket Status: ${wsStatus}`}>
                <Chip
                  icon={<WsIcon sx={{ fontSize: '14px !important' }} />}
                  label={wsStatus}
                  color={wsStatus === 'CONNECTED' ? 'success' : 'error'}
                  variant="outlined"
                  size="small"
                  sx={{ border: 'none', '& .MuiChip-label': { fontSize: '11px', fontWeight: 600 } }}
                />
              </Tooltip>

              <Tooltip title="Database Connection Status: ONLINE">
                <Chip
                  icon={<DbIcon sx={{ fontSize: '14px !important' }} />}
                  label="DB: ONLINE"
                  color="success"
                  variant="outlined"
                  size="small"
                  sx={{ border: 'none', '& .MuiChip-label': { fontSize: '11px', fontWeight: 600 } }}
                />
              </Tooltip>

              <Tooltip title="GPU Core Acceleration: ENABLED">
                <Chip
                  icon={<GpuIcon sx={{ fontSize: '14px !important' }} />}
                  label="CUDA: ONLINE"
                  color="success"
                  variant="outlined"
                  size="small"
                  sx={{ border: 'none', '& .MuiChip-label': { fontSize: '11px', fontWeight: 600 } }}
                />
              </Tooltip>
            </Box>

            <Divider orientation="vertical" flexItem sx={{ display: { xs: 'none', md: 'block' }, mx: 0.5, borderColor: 'rgba(148, 163, 184, 0.15)' }} />

            {/* Notification Bell */}
            <IconButton
              color="inherit"
              onClick={() => {
                resetUnreadCount();
                navigate('/alerts');
              }}
            >
              <Badge badgeContent={unreadCount} color="error" max={99}>
                <AlertsIcon />
              </Badge>
            </IconButton>

            {/* Profile badge and user identity */}
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 1,
                cursor: 'pointer',
                p: 0.5,
                borderRadius: '8px',
                '&:hover': { bgcolor: 'rgba(148, 163, 184, 0.08)' }
              }}
              onClick={handleProfileMenuOpen}
            >
              <AccountCircle />
              <Box sx={{ display: { xs: 'none', sm: 'block' }, textAlign: 'left' }}>
                <Typography variant="body1" sx={{ fontWeight: 600, color: 'text.primary', leading: 1 }}>
                  {username || 'Operator'}
                </Typography>
                <Typography variant="body2" sx={{ color: 'text.secondary', fontSize: '10px', leading: 1 }}>
                  {role || 'Investigator'}
                </Typography>
              </Box>
            </Box>

            <Menu
              anchorEl={anchorEl}
              open={Boolean(anchorEl)}
              onClose={handleMenuClose}
              slotProps={{
                paper: {
                  sx: {
                    backgroundColor: 'background.paper',
                    border: '1px solid rgba(26, 37, 60, 0.6)',
                    backdropFilter: 'blur(12px)',
                    boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
                    mt: 1.5,
                    minWidth: 150
                  }
                }
              }}
            >
              <MenuItem onClick={handleLogout} sx={{ color: 'error.main' }}>
                <ListItemIcon sx={{ color: 'error.main' }}>
                  <LogoutIcon fontSize="small" />
                </ListItemIcon>
                <ListItemText primary="Log Out" />
              </MenuItem>
            </Menu>
          </Box>
        </Toolbar>
      </AppBar>

      {/* Left Sidebar Drawer */}
      <Drawer
        variant="permanent"
        open={open}
        slotProps={{
          paper: {
            sx: {
              backgroundColor: 'rgba(5, 7, 13, 0.95)',
              borderRight: '1px solid rgba(26, 37, 60, 0.4)',
              width: open ? drawerWidth : 60,
              overflowX: 'hidden',
              transition: (theme: any) =>
                theme.transitions.create('width', {
                  easing: theme.transitions.easing.sharp,
                  duration: theme.transitions.duration.enteringScreen,
                }),
              ...(!open && {
                transition: (theme: any) =>
                  theme.transitions.create('width', {
                    easing: theme.transitions.easing.sharp,
                    duration: theme.transitions.duration.leavingScreen,
                  }),
              }),
            }
          }
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', px: 2, py: 1.5 }}>
          {open && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Box
                component="div"
                sx={{
                  width: 24,
                  height: 24,
                  borderRadius: '6px',
                  background: 'linear-gradient(135deg, #00A8FF 0%, #0075B3 100%)',
                }}
              />
              <Typography variant="h5" sx={{ fontWeight: 800, color: 'primary.main', letterSpacing: '-0.03em' }}>
                VisionGuard
              </Typography>
            </Box>
          )}
          <IconButton onClick={handleDrawerClose} sx={{ ...(!open && { display: 'none' }) }}>
            <ChevronLeftIcon />
          </IconButton>
          <IconButton
            onClick={handleDrawerOpen}
            sx={{
              margin: 'auto',
              ...(open && { display: 'none' })
            }}
          >
            <MenuIcon />
          </IconButton>
        </Box>
        <Divider sx={{ borderColor: 'rgba(148, 163, 184, 0.15)' }} />

        {/* Sidebar list items */}
        <List sx={{ mt: 1 }}>
          {menuItems.map((item) => {
            const isSelected = location.pathname === item.path;
            return (
              <Tooltip key={item.text} title={!open ? item.text : ''} placement="right">
                <ListItemButton
                  component={Link}
                  to={item.path}
                  selected={isSelected}
                  sx={{
                    justifyContent: open ? 'initial' : 'center',
                    px: 2.5,
                  }}
                >
                  <ListItemIcon
                    sx={{
                      minWidth: 0,
                      mr: open ? 2 : 'auto',
                      justifyContent: 'center',
                      color: isSelected ? 'primary.main' : 'text.secondary',
                    }}
                  >
                    {item.icon}
                  </ListItemIcon>
                  {open && (
                    <ListItemText
                      primary={
                        <Typography
                          sx={{
                            fontSize: '13px',
                            fontWeight: isSelected ? 600 : 500,
                          }}
                        >
                          {item.text}
                        </Typography>
                      }
                    />
                  )}
                </ListItemButton>
              </Tooltip>
            );
          })}
        </List>
      </Drawer>

      {/* Main Content Grid Area */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          pt: 10,
          width: `calc(100% - ${open ? drawerWidth : 60}px)`,
          transition: (theme) =>
            theme.transitions.create('width', {
              easing: theme.transitions.easing.sharp,
              duration: theme.transitions.duration.enteringScreen,
            }),
          bgcolor: 'background.default',
        }}
      >
        <Outlet />
      </Box>
    </Box>
  );
};

export default MainLayout;
