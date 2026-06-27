import { createTheme } from '@mui/material/styles';

export const socTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00A8FF', // Active blue
      light: '#33BAFF',
      dark: '#0075B3',
      contrastText: '#FFFFFF',
    },
    secondary: {
      main: '#90CAF9',
    },
    background: {
      default: '#05070D', // Slate-black backdrop
      paper: 'rgba(16, 26, 48, 0.65)', // Glass backdrop paper card
    },
    text: {
      primary: '#E2E8F0', // Soft white/grey
      secondary: '#94A3B8', // Slate grey
    },
    success: {
      main: '#00E676',
    },
    warning: {
      main: '#FF9100',
    },
    error: {
      main: '#FF1744', // Critical red
    },
    divider: 'rgba(148, 163, 184, 0.12)',
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 700,
      letterSpacing: '-0.025em',
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 600,
      letterSpacing: '-0.02em',
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 600,
      letterSpacing: '-0.015em',
    },
    h6: {
      fontSize: '1rem',
      fontWeight: 600,
      letterSpacing: '-0.01em',
    },
    body1: {
      fontSize: '0.875rem',
      lineHeight: 1.5,
    },
    body2: {
      fontSize: '0.75rem',
      lineHeight: 1.43,
    },
    button: {
      textTransform: 'none',
      fontWeight: 600,
    },
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          backgroundColor: '#05070D',
          color: '#E2E8F0',
          scrollbarWidth: 'thin',
          '&::-webkit-scrollbar': {
            width: '6px',
            height: '6px',
          },
          '&::-webkit-scrollbar-track': {
            background: '#05070D',
          },
          '&::-webkit-scrollbar-thumb': {
            background: '#1A253C',
            borderRadius: '3px',
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundColor: 'rgba(16, 26, 48, 0.65)',
          backdropFilter: 'blur(12px)',
          WebkitBackdropFilter: 'blur(12px)',
          border: '1px solid rgba(26, 37, 60, 0.6)',
          borderRadius: '12px',
          boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
          backgroundImage: 'none',
          transition: 'transform 0.2s ease-in-out, border-color 0.2s ease-in-out, box-shadow 0.2s ease-in-out',
          '&:hover': {
            borderColor: 'rgba(0, 168, 255, 0.4)',
            boxShadow: '0 8px 32px 0 rgba(0, 168, 255, 0.08)',
          },
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: '8px',
          boxShadow: 'none',
          '&:hover': {
            boxShadow: 'none',
          },
        },
      },
      variants: [
        {
          props: { variant: 'contained', color: 'primary' },
          style: {
            background: 'linear-gradient(135deg, #00A8FF 0%, #0075B3 100%)',
            color: '#FFFFFF',
            '&:hover': {
              background: 'linear-gradient(135deg, #33BAFF 0%, #008AD6 100%)',
            },
          },
        },
      ],
    },
    MuiTableHead: {
      styleOverrides: {
        root: {
          backgroundColor: 'rgba(10, 15, 30, 0.8)',
          '& .MuiTableCell-head': {
            color: '#94A3B8',
            fontWeight: 600,
            borderBottom: '2px solid rgba(26, 37, 60, 0.8)',
          },
        },
      },
    },
    MuiTableCell: {
      styleOverrides: {
        root: {
          borderBottom: '1px solid rgba(26, 37, 60, 0.4)',
          padding: '12px 16px',
        },
      },
    },
    MuiListItemButton: {
      styleOverrides: {
        root: {
          borderRadius: '8px',
          margin: '4px 8px',
          padding: '8px 12px',
          transition: 'background-color 0.2s ease-in-out, color 0.2s ease-in-out',
          '&.Mui-selected': {
            backgroundColor: 'rgba(0, 168, 255, 0.15)',
            color: '#00A8FF',
            '& .MuiListItemIcon-root': {
              color: '#00A8FF',
            },
            '&:hover': {
              backgroundColor: 'rgba(0, 168, 255, 0.2)',
            },
          },
          '&:hover': {
            backgroundColor: 'rgba(148, 163, 184, 0.08)',
          },
        },
      },
    },
  },
});
